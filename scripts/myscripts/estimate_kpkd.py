#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线估计每个关节的 Kp/Kd（仅计算，不参与控制）。

订阅：
- /debug/effort_minus_gravity (sensor_msgs/JointState): effort = tau_measured - tau_gravity  -> 记为 y
- /joint_states 或用户指定话题 (sensor_msgs/JointState): position/velocity -> q, dq
- /joint_cmd_mirror/joint_states (sensor_msgs/JointState): position -> q_des（来自 viewer 的镜像）

输出：
- 周期性在控制台打印每关节 {Kp, Kd}（可选带摩擦参数）
- /est_kpkd/<joint>/kp, /est_kpkd/<joint>/kd (std_msgs/Float64)
- 可选 --with-friction 估计 Fc, B，并发布到 /est_kpkd/<joint>/{fc,b}

使用：
  python scripts/myscripts/estimate_kpkd.py \
    --state-topic /joint_states \
    --y-topic /debug/effort_minus_gravity \
    --qdes-topic /joint_cmd_mirror/joint_states \
    --rate 10 \
    --window 5.0 \
    --l2 0.0 \
    [--with-friction]

说明：
- 本脚本只做数据拟合，不改变任何控制；若 q_des 源不存在，估计将退化（只能估计 Kd 或非常不稳定）。
- 请确保三个话题时间戳尽量一致；脚本做了最近邻对齐（容差 default 50ms）。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64
except Exception as e:  # pragma: no cover - 环境依赖
    print(f"[ERR] 需要 ROS2 环境 (rclpy)：{e}")
    raise


@dataclass
class Sample:
    t: float
    # 对齐后的序列，均为列表，与 joint_names 对应
    e: List[float]   # q_des - q
    v: List[float]   # dq
    y: List[float]   # tau_meas_minus_gravity


class Estimator(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('estimate_kpkd')
        self.args = args
        self.rate_hz = float(args.rate)
        self.window_s = float(args.window)
        self.l2 = float(args.l2)
        self.with_friction = bool(args.with_friction)
        self.sync_tol = float(args.sync_tol)
        self.hold_qdes_sec = float(getattr(args, 'hold_qdes_sec', 0.0))
        self.exact_fit = bool(getattr(args, 'exact_fit', False))
        # 稳态器/鲁棒项
        self.emin = float(args.emin)
        self.vmin = float(args.vmin)
        self.max_cond = float(args.max_cond)
        self.min_samples = int(args.min_samples)
        self.mad_k = float(args.mad_k)
        self.ema = float(args.ema)
        self.why_empty = bool(getattr(args, 'why_empty', False))
        self._last_stat_log: float = 0.0
        self._stats = {
            'waiting_missing': 0,
            'time_mismatch': 0,
            'align_empty': 0,
            'window_short': 0,
            'solved_joints': 0,
            'skipped_joints_few_mask': 0,
            'skipped_joints_bad_cond': 0,
            'exact_pairs_tried': 0,
            'exact_pairs_used': 0,
        }
        self._last_dt_dbg: Optional[Tuple[float, float, float]] = None

        # 缓存最近的三类 JointState（按 name->(t,vec) 存最后一帧）
        self._last_state: Optional[Tuple[float, List[str], List[float], List[float]]] = None
        self._last_y: Optional[Tuple[float, List[str], List[float]]] = None
        self._last_qdes: Optional[Tuple[float, List[str], List[float]]] = None

        # 统一的关节名序（按第一次看到的 state 为准）
        self.joint_names: Optional[List[str]] = None
        self.name_to_idx: Dict[str, int] = {}

        # 环形窗口（滑动时间窗）
        self.samples: Deque[Sample] = collections.deque(maxlen=100000)

        # 发布器：每关节 kp/kd（可选 fc/b）
        self.pub_map: Dict[str, Dict[str, any]] = {}

        # 订阅
        qos = 10
        self.sub_state = self.create_subscription(JointState, args.state_topic, self._cb_state, qos)
        self.sub_y = self.create_subscription(JointState, args.y_topic, self._cb_y, qos)
        self.sub_qdes = self.create_subscription(JointState, args.qdes_topic, self._cb_qdes, qos)

        # 定时估计
        period = 1.0 / max(1e-3, self.rate_hz)
        self.timer = self.create_timer(period, self._on_timer)
        # EMA 历史
        self.prev_theta: Dict[str, Dict[str, float]] = {}
        self.get_logger().info(
            f"Estimator started: state={args.state_topic}, y={args.y_topic}, qdes={args.qdes_topic}, "
            f"rate={self.rate_hz}Hz, window={self.window_s}s, l2={self.l2}, with_friction={self.with_friction}, tol={self.sync_tol}s, "
            f"hold_qdes={self.hold_qdes_sec}s, emin={self.emin}, vmin={self.vmin}, min_samples={self.min_samples}, max_cond={self.max_cond}, "
            f"ema={self.ema}, mad_k={self.mad_k}, why_empty={self.why_empty}, exact_fit={self.exact_fit}"
        )

    # ---- Callbacks ----
    def _cb_state(self, msg: JointState):
        t = self._stamp_to_sec(msg.header.stamp)
        names = list(msg.name)
        q = list(msg.position)
        dq = list(msg.velocity)
        if self.joint_names is None:
            self.joint_names = names
            self.name_to_idx = {n: i for i, n in enumerate(self.joint_names)}
        self._last_state = (t, names, q, dq)

    def _cb_y(self, msg: JointState):
        t = self._stamp_to_sec(msg.header.stamp)
        names = list(msg.name)
        y = list(msg.effort)
        self._last_y = (t, names, y)

    def _cb_qdes(self, msg: JointState):
        t = self._stamp_to_sec(msg.header.stamp)
        names = list(msg.name)
        qd = list(msg.position)
        self._last_qdes = (t, names, qd)

    # ---- Core ----
    def _on_timer(self):
        # 需要三者都有
        if self._last_state is None or self._last_y is None or self._last_qdes is None:
            self._stats['waiting_missing'] += 1
            self._maybe_log_empty_reason()
            return
        ts, ns, q, dq = self._last_state
        ty, ny, y = self._last_y
        td, nd, qdes = self._last_qdes
        # 时间同步容差（最近邻）
        t_now = max(ts, ty, td)
        dt_s = abs(t_now - ts)
        dt_y = abs(t_now - ty)
        dt_d = abs(t_now - td)
        allow_qdes_hold = (dt_s <= self.sync_tol) and (dt_y <= self.sync_tol) and (dt_d <= max(self.sync_tol, self.hold_qdes_sec))
        if not ((dt_s <= self.sync_tol and dt_y <= self.sync_tol and dt_d <= self.sync_tol) or allow_qdes_hold):
            self._stats['time_mismatch'] += 1
            self._last_dt_dbg = (dt_s, dt_y, dt_d)
            self._maybe_log_empty_reason()
            return
        # 对齐顺序：以 self.joint_names 为准
        if self.joint_names is None:
            self.joint_names = ns
            self.name_to_idx = {n: i for i, n in enumerate(self.joint_names)}
        e_vec, v_vec, y_vec = self._align_vectors(qdes, nd, q, ns, dq, ns, y, ny)
        if e_vec is None:
            self._stats['align_empty'] += 1
            self._maybe_log_empty_reason()
            return
        self.samples.append(Sample(t_now, e_vec, v_vec, y_vec))
        self._drop_old_samples()
        # 窗口足够才估计
        if len(self.samples) < max(10, int(self.rate_hz * min(1.0, self.window_s))):
            self._stats['window_short'] += 1
            self._maybe_log_empty_reason()
            return
        theta_map = self._estimate_window()
        self._publish_and_log(theta_map)

    # ---- Helpers ----
    def _align_vectors(self, qdes, nd, q, ns, dq, ns2, y, ny) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
        try:
            name_to_qdes = {n: qdes[i] for i, n in enumerate(nd)}
            name_to_q = {n: q[i] for i, n in enumerate(ns)}
            name_to_dq = {n: dq[i] for i, n in enumerate(ns2)}
            name_to_y = {n: y[i] for i, n in enumerate(ny)}
            e_vec, v_vec, y_vec = [], [], []
            for n in self.joint_names:
                if n in name_to_q and n in name_to_dq and n in name_to_qdes and n in name_to_y:
                    e_vec.append(float(name_to_qdes[n] - name_to_q[n]))
                    v_vec.append(float(name_to_dq[n]))
                    y_vec.append(float(name_to_y[n]))
            if not e_vec:
                return None, None, None
            return e_vec, v_vec, y_vec
        except Exception:
            return None, None, None

    def _drop_old_samples(self):
        if not self.samples:
            return
        t_latest = self.samples[-1].t
        horizon = t_latest - self.window_s
        while self.samples and self.samples[0].t < horizon:
            self.samples.popleft()

    def _estimate_window(self) -> Dict[str, Dict[str, float]]:
        # 堆叠窗口数据
        names = self.joint_names
        E, V, Y = [], [], []
        for s in self.samples:
            E.append(s.e)
            V.append(s.v)
            Y.append(s.y)
        E = np.asarray(E)
        V = np.asarray(V)
        Y = np.asarray(Y)

        theta_map: Dict[str, Dict[str, float]] = {}
        solved = 0
        for j, n in enumerate(names):
            e = E[:, j:j+1]
            v = V[:, j:j+1]
            y = Y[:, j:j+1]
            mask = (np.abs(e) >= self.emin) | (np.abs(v) >= self.vmin)
            mask_flat = np.squeeze(mask)
            if mask_flat.ndim != 1:
                mask_flat = mask_flat.reshape(-1)

            p = 4 if self.with_friction else 2
            if self.exact_fit:
                mcount = int(mask_flat.sum())
                if mcount < p:
                    # 单样本降阶精确解（最小范数）：优先用 e 解 Kp，否则用 v 解 Kd
                    if mcount == 1 and not self.with_friction:
                        idx = int(np.where(mask_flat)[0][0])
                        e1 = float(e[idx, 0])
                        v1 = float(v[idx, 0])
                        y1 = float(y[idx, 0])
                        if abs(e1) >= self.emin:
                            Kp = y1 / e1
                            Kd = 0.0
                            theta_map[n] = {"Kp": float(Kp), "Kd": float(Kd)}
                            solved += 1
                            continue
                        elif abs(v1) >= self.vmin:
                            Kp = 0.0
                            Kd = -y1 / v1
                            theta_map[n] = {"Kp": float(Kp), "Kd": float(Kd)}
                            solved += 1
                            continue
                    self._stats['skipped_joints_few_mask'] += 1
                    continue
                idx_all = np.where(mask_flat)[0]
                solved_here = False
                # 尝试最近的 p 条以及向前滑动一定范围
                start_min = max(0, len(idx_all) - p - 20)
                for start in range(start_min, len(idx_all) - p + 1):
                    use = idx_all[start:start + p]
                    e_m = e[use, :]
                    v_m = v[use, :]
                    y_m = y[use, :]
                    if self.with_friction:
                        Phi = np.hstack([e_m, -v_m, np.sign(v_m), v_m])
                        names_theta = ["Kp", "Kd", "Fc", "B"]
                    else:
                        Phi = np.hstack([e_m, -v_m])
                        names_theta = ["Kp", "Kd"]
                    try:
                        cond = np.linalg.cond(Phi)
                        if np.isnan(cond) or np.isinf(cond) or cond > self.max_cond:
                            continue
                        theta = np.linalg.solve(Phi, y_m)
                        if np.max(np.abs(Phi @ theta - y_m)) <= 1e-9:
                            theta = theta.flatten().tolist()
                            theta_map[n] = {k: float(v) for k, v in zip(names_theta, theta)}
                            solved += 1
                            solved_here = True
                            break
                    except Exception:
                        continue
                if not solved_here:
                    self._stats['skipped_joints_bad_cond'] += 1
                    continue
            else:
                if mask_flat.sum() < self.min_samples:
                    self._stats['skipped_joints_few_mask'] += 1
                    continue
                e_m = e[mask_flat, :]
                v_m = v[mask_flat, :]
                y_m = y[mask_flat, :]
                if self.with_friction:
                    Phi = np.hstack([e_m, -v_m, np.sign(v_m), v_m])
                    names_theta = ["Kp", "Kd", "Fc", "B"]
                else:
                    Phi = np.hstack([e_m, -v_m])
                    names_theta = ["Kp", "Kd"]
                I = np.eye(Phi.shape[1])
                try:
                    gram = Phi.T @ Phi + self.l2 * I
                    cond = np.linalg.cond(gram)
                    if np.isnan(cond) or np.isinf(cond) or cond > self.max_cond:
                        self._stats['skipped_joints_bad_cond'] += 1
                        continue
                    theta = np.linalg.pinv(gram) @ (Phi.T @ y_m)
                except Exception:
                    continue
                try:
                    y_pred = Phi @ theta
                    r = (y_m - y_pred).reshape(-1)
                    mad = np.median(np.abs(r - np.median(r))) + 1e-9
                    good = np.abs(r) <= self.mad_k * 1.4826 * mad
                    if good.sum() >= self.min_samples and good.sum() < len(r):
                        Phi2 = Phi[good, :]
                        y2 = y_m[good, :]
                        gram2 = Phi2.T @ Phi2 + self.l2 * I
                        cond2 = np.linalg.cond(gram2)
                        if not (np.isnan(cond2) or np.isinf(cond2) or cond2 > self.max_cond):
                            theta = np.linalg.pinv(gram2) @ (Phi2.T @ y2)
                except Exception:
                    pass
                theta = theta.flatten().tolist()
                theta_map[n] = {k: float(v) for k, v in zip(names_theta, theta)}
                solved += 1
        self._stats['solved_joints'] += solved
        return theta_map

    def _publish_and_log(self, theta_map: Dict[str, Dict[str, float]]):
        if not theta_map:
            self._maybe_log_empty_reason()
        # 首次为关节建发布器
        for n in theta_map.keys():
            if n not in self.pub_map:
                self.pub_map[n] = {}
                self.pub_map[n]['kp'] = self.create_publisher(Float64, f"/est_kpkd/{n}/kp", 10)
                self.pub_map[n]['kd'] = self.create_publisher(Float64, f"/est_kpkd/{n}/kd", 10)
                if self.with_friction:
                    self.pub_map[n]['fc'] = self.create_publisher(Float64, f"/est_kpkd/{n}/fc", 10)
                    self.pub_map[n]['b'] = self.create_publisher(Float64, f"/est_kpkd/{n}/b", 10)
        # EMA 平滑并发布
        for n, d in theta_map.items():
            if self.ema > 0.0 and n in self.prev_theta:
                prev = self.prev_theta[n]
                smoothed = {}
                for k, raw in d.items():
                    pr = float(prev.get(k, raw))
                    smoothed[k] = float(self.ema * float(raw) + (1.0 - self.ema) * pr)
                d_use = smoothed
            else:
                d_use = d
            try:
                if 'kp' in self.pub_map[n]:
                    m = Float64(); m.data = float(d_use.get('Kp', 0.0)); self.pub_map[n]['kp'].publish(m)
                if 'kd' in self.pub_map[n]:
                    m = Float64(); m.data = float(d_use.get('Kd', 0.0)); self.pub_map[n]['kd'].publish(m)
                if self.with_friction:
                    m = Float64(); m.data = float(d_use.get('Fc', 0.0)); self.pub_map[n]['fc'].publish(m)
                    m = Float64(); m.data = float(d_use.get('B', 0.0)); self.pub_map[n]['b'].publish(m)
            except Exception:
                pass
            self.prev_theta[n] = d_use
        try:
            sample = {n: {k: round(v, 4) for k, v in d.items()} for n, d in list(self.prev_theta.items())[:8]}
            self.get_logger().info("est(theta) sample=" + json.dumps(sample, ensure_ascii=False))
        except Exception:
            pass

    def _maybe_log_empty_reason(self):
        if not self.why_empty:
            return
        now = time.time()
        if now - self._last_stat_log < 2.0:
            return
        self._last_stat_log = now
        dt_dbg = self._last_dt_dbg
        dt_msg = f" dt(s,y,d)={tuple(round(x, 4) for x in dt_dbg)}" if dt_dbg else ""
        self.get_logger().info(
            f"why-empty: missing={self._stats['waiting_missing']}, time_mismatch={self._stats['time_mismatch']}, "
            f"align_empty={self._stats['align_empty']}, window_short={self._stats['window_short']}, "
            f"skipped_few_mask={self._stats['skipped_joints_few_mask']}, skipped_bad_cond={self._stats['skipped_joints_bad_cond']}, "
            f"solved_total={self._stats['solved_joints']}.{dt_msg}"
        )

    # ---- Helpers ----
    def _align_vectors(self, qdes, nd, q, ns, dq, ns2, y, ny) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
        try:
            name_to_qdes = {n: qdes[i] for i, n in enumerate(nd)}
            name_to_q = {n: q[i] for i, n in enumerate(ns)}
            name_to_dq = {n: dq[i] for i, n in enumerate(ns2)}
            name_to_y = {n: y[i] for i, n in enumerate(ny)}
            e_vec, v_vec, y_vec = [], [], []
            for n in self.joint_names:
                if n in name_to_q and n in name_to_dq and n in name_to_qdes and n in name_to_y:
                    e_vec.append(float(name_to_qdes[n] - name_to_q[n]))
                    v_vec.append(float(name_to_dq[n]))
                    y_vec.append(float(name_to_y[n]))
            if not e_vec:
                return None, None, None
            return e_vec, v_vec, y_vec
        except Exception:
            return None, None, None

    def _drop_old_samples(self):
        if not self.samples:
            return
        t_latest = self.samples[-1].t
        horizon = t_latest - self.window_s
        while self.samples and self.samples[0].t < horizon:
            self.samples.popleft()

    def _estimate_window(self) -> Dict[str, Dict[str, float]]:
        # 堆叠窗口数据
        names = self.joint_names
        dim = len(names)
        E = []  # [T, dim]
        V = []  # [T, dim]
        Y = []  # [T, dim]
        for s in self.samples:
            E.append(s.e)
            V.append(s.v)
            Y.append(s.y)
        E = np.asarray(E)  # [T, D]
        V = np.asarray(V)
        Y = np.asarray(Y)
        # 对每个关节独立拟合
        theta_map: Dict[str, Dict[str, float]] = {}
        solved = 0
        for j, n in enumerate(names):
            e = E[:, j:j+1]  # [T,1]
            v = V[:, j:j+1]
            y = Y[:, j:j+1]
            # 门限：剔除 |e| 和 |v| 都过小的样本，避免病态
            mask = (np.abs(e) >= self.emin) | (np.abs(v) >= self.vmin)
            # 将 (T,1) 的布尔掩码压成 (T,) 并只在样本维度筛选，保持列向量形状
            mask_flat = np.squeeze(mask)
            if mask_flat.ndim != 1:
                mask_flat = mask_flat.reshape(-1)
            # 精确拟合：直接用两条样本解 2×2（仅 Kp/Kd）
            if self.exact_fit and not self.with_friction:
                idxs = np.where(mask_flat)[0]
                mcount = int(idxs.size)
                if mcount == 0:
                    self._stats['skipped_joints_few_mask'] += 1
                    continue
                if mcount == 1:
                    # 单条样本的最小范数解：优先用 e 解 Kp，否则用 v 解 Kd
                    i = int(idxs[0])
                    e1 = float(e[i, 0]); v1 = float(v[i, 0]); y1 = float(y[i, 0])
                    if abs(e1) >= self.emin:
                        theta_map[n] = {"Kp": float(y1 / e1), "Kd": 0.0}
                        solved += 1
                        continue
                    if abs(v1) >= self.vmin:
                        theta_map[n] = {"Kp": 0.0, "Kd": float(-y1 / v1)}
                        solved += 1
                        continue
                    self._stats['skipped_joints_few_mask'] += 1
                    continue
                # 至少两条：从最近的若干条中选行列式最大的两条，避免奇异
                use_pool = idxs[max(0, len(idxs) - 40):]  # 最近 40 条以内搜索
                best_det = 0.0; best_pair = None
                for a_idx in range(len(use_pool)):
                    ia = int(use_pool[a_idx])
                    ea = float(e[ia, 0]); va = float(v[ia, 0])
                    for b_idx in range(a_idx + 1, len(use_pool)):
                        ib = int(use_pool[b_idx])
                        eb = float(e[ib, 0]); vb = float(v[ib, 0])
                        det = abs(ea * (-vb) - eb * (-va))
                        self._stats['exact_pairs_tried'] += 1
                        if det > best_det:
                            best_det = det; best_pair = (ia, ib)
                if best_pair is None or best_det < 1e-12:
                    self._stats['skipped_joints_bad_cond'] += 1
                    continue
                ia, ib = best_pair
                A = np.array([[float(e[ia, 0]), -float(v[ia, 0])],
                              [float(e[ib, 0]), -float(v[ib, 0])]], dtype=float)
                y2 = np.array([float(y[ia, 0]), float(y[ib, 0])], dtype=float)
                try:
                    theta = np.linalg.solve(A, y2)
                except Exception:
                    self._stats['skipped_joints_bad_cond'] += 1
                    continue
                theta_map[n] = {"Kp": float(theta[0]), "Kd": float(theta[1])}
                solved += 1
                self._stats['exact_pairs_used'] += 1
                continue
            # 普通（最小二乘 / 含摩擦）
            if mask_flat.sum() < self.min_samples:
                self._stats['skipped_joints_few_mask'] += 1
                continue
            e_m = e[mask_flat, :]
            v_m = v[mask_flat, :]
            y_m = y[mask_flat, :]
            if self.with_friction:
                Phi = np.hstack([e_m, -v_m, np.sign(v_m), v_m])  # [Kp, Kd, Fc, B]
                names_theta = ["Kp", "Kd", "Fc", "B"]
            else:
                Phi = np.hstack([e_m, -v_m])                # [Kp, Kd]
                names_theta = ["Kp", "Kd"]
            I = np.eye(Phi.shape[1])
            # 条件数守卫
            try:
                gram = Phi.T @ Phi + self.l2 * I
                cond = np.linalg.cond(gram)
                if np.isnan(cond) or np.isinf(cond) or cond > self.max_cond:
                    self._stats['skipped_joints_bad_cond'] += 1
                    continue
                theta = np.linalg.pinv(gram) @ (Phi.T @ y_m)
            except Exception:
                continue
            # 一次 MAD 基的离群点剔除（可选）
            try:
                y_pred = Phi @ theta
                r = (y_m - y_pred).reshape(-1)
                mad = np.median(np.abs(r - np.median(r))) + 1e-9
                good = np.abs(r) <= self.mad_k * 1.4826 * mad  # 1.4826 ~ sigma≈MAD
                if good.sum() >= self.min_samples and good.sum() < len(r):
                    # good 是一维样本掩码，对行筛选
                    Phi2 = Phi[good, :]
                    y2 = y_m[mask_flat, :][good, :]
                    gram2 = Phi2.T @ Phi2 + self.l2 * I
                    cond2 = np.linalg.cond(gram2)
                    if not (np.isnan(cond2) or np.isinf(cond2) or cond2 > self.max_cond):
                        theta = np.linalg.pinv(gram2) @ (Phi2.T @ y2)
            except Exception:
                pass
            theta = theta.flatten().tolist()
            theta_map[n] = {k: float(v) for k, v in zip(names_theta, theta)}
            solved += 1
        self._stats['solved_joints'] += solved
        return theta_map

    def _publish_and_log(self, theta_map: Dict[str, Dict[str, float]]):
        if not theta_map:
            self._maybe_log_empty_reason()
        # 首次为关节建发布器
        for n in theta_map.keys():
            if n not in self.pub_map:
                self.pub_map[n] = {}
                self.pub_map[n]['kp'] = self.create_publisher(Float64, f"/est_kpkd/{n}/kp", 10)
                self.pub_map[n]['kd'] = self.create_publisher(Float64, f"/est_kpkd/{n}/kd", 10)
                if self.with_friction:
                    self.pub_map[n]['fc'] = self.create_publisher(Float64, f"/est_kpkd/{n}/fc", 10)
                    self.pub_map[n]['b'] = self.create_publisher(Float64, f"/est_kpkd/{n}/b", 10)
        # EMA 平滑并发布
        for n, d in theta_map.items():
            # EMA: new = ema*raw + (1-ema)*prev
            if self.ema > 0.0 and n in self.prev_theta:
                prev = self.prev_theta[n]
                smoothed = {}
                for k, raw in d.items():
                    pr = float(prev.get(k, raw))
                    smoothed[k] = float(self.ema * float(raw) + (1.0 - self.ema) * pr)
                d_use = smoothed
            else:
                d_use = d
            try:
                if 'kp' in self.pub_map[n]:
                    m = Float64(); m.data = float(d_use.get('Kp', 0.0)); self.pub_map[n]['kp'].publish(m)
                if 'kd' in self.pub_map[n]:
                    m = Float64(); m.data = float(d_use.get('Kd', 0.0)); self.pub_map[n]['kd'].publish(m)
                if self.with_friction:
                    m = Float64(); m.data = float(d_use.get('Fc', 0.0)); self.pub_map[n]['fc'].publish(m)
                    m = Float64(); m.data = float(d_use.get('B', 0.0)); self.pub_map[n]['b'].publish(m)
            except Exception:
                pass
            # 保存 EMA 历史
            self.prev_theta[n] = d_use
        # 日志（简略）
        try:
            sample = {n: {k: round(v, 4) for k, v in d.items()} for n, d in list(self.prev_theta.items())[:8]}
            self.get_logger().info("est(theta) sample=" + json.dumps(sample, ensure_ascii=False))
        except Exception:
            pass

    def _maybe_log_empty_reason(self):
        if not self.why_empty:
            return
        now = time.time()
        if now - self._last_stat_log < 2.0:
            return
        self._last_stat_log = now
        dt_dbg = self._last_dt_dbg
        dt_msg = f" dt(s,y,d)={tuple(round(x, 4) for x in dt_dbg)}" if dt_dbg else ""
        self.get_logger().info(
            f"why-empty: missing={self._stats['waiting_missing']}, time_mismatch={self._stats['time_mismatch']}, "
            f"align_empty={self._stats['align_empty']}, window_short={self._stats['window_short']}, "
            f"skipped_few_mask={self._stats['skipped_joints_few_mask']}, skipped_bad_cond={self._stats['skipped_joints_bad_cond']}, "
            f"solved_total={self._stats['solved_joints']}.{dt_msg}"
        )

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        try:
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except Exception:
            return time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state-topic', default='/joint_states', help='机器人状态 JointState 话题（position/velocity）')
    ap.add_argument('--y-topic', default='/debug/effort_minus_gravity', help='JointState.effort= tau_meas - gravity')
    ap.add_argument('--qdes-topic', default='/joint_cmd_mirror/joint_states', help='JointState.position = 期望角 (来自 viewer 镜像)')
    ap.add_argument('--rate', type=float, default=10.0, help='估计频率 Hz')
    ap.add_argument('--window', type=float, default=5.0, help='滑动窗口长度 s')
    ap.add_argument('--l2', type=float, default=0.0, help='岭回归系数')
    ap.add_argument('--with-friction', action='store_true', help='同时估计 Fc, B')
    ap.add_argument('--sync-tol', type=float, default=0.2, help='三路话题时间对齐容差 s (默认 200ms)')
    ap.add_argument('--hold-qdes-sec', type=float, default=0.5, help='q_des 慢话题允许的持有时间 s（仅用于时间对齐）')
    ap.add_argument('--why-empty', action='store_true', help='打印“为何没有输出”的调试统计（节流 2s）')
    ap.add_argument('--exact-fit', action='store_true', help='精确拟合模式：使用最近 p 个样本解方程，使 y==Kp*e-Kd*v(+fric) 在样本上严格成立')
    # 稳定化参数
    ap.add_argument('--emin', type=float, default=1e-3, help='|e| 的最小门限（小于该值的样本将被过滤）')
    ap.add_argument('--vmin', type=float, default=1e-3, help='|dq| 的最小门限（小于该值的样本将被过滤）')
    ap.add_argument('--min-samples', type=int, default=10, help='每个关节最少用于拟合的样本数量')
    ap.add_argument('--max-cond', type=float, default=1e6, help='Gram 矩阵条件数上限，超出则跳过拟合')
    ap.add_argument('--ema', type=float, default=0.6, help='参数 EMA 平滑系数，0=不用，1=只信任新值')
    ap.add_argument('--mad-k', type=float, default=3.5, help='MAD 离群点阈值倍率（3~4 合理）')
    args = ap.parse_args()

    rclpy.init(args=None)
    node = Estimator(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
