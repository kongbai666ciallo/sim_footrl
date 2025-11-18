#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 关节命令发布器

向指定话题发布关节命令，消息类型为 std_msgs/String，内容为 JSON 字符串：
[["joint_name", angle], ["joint_name2", angle2], ...]

用法示例：
  # 以角度(度)发送两个关节命令，一次性发布
  python scripts/test_publisher.py \
    --topic /joint_cmd_array \
    --unit deg \
    --pairs "l_elbow_joint:30,r_wrist_joint:-15" \
    --oneshot

  # 以弧度发送并以 2Hz 周期发布
  python scripts/test_publisher.py --unit rad --pairs "l_elbow_joint:0.5" --rate 2.0

    # 对命令关节做正弦摆（以度为单位，幅度 10 度，频率 1Hz，围绕给定基准角）
    python scripts/test_publisher.py \
        --topic /joint_cmd_array \
        --unit deg \
        --pairs "l_elbow_joint:30,r_wrist_joint:-15" \
        --rate 100 \
        --sine-amp 10 --sine-freq-hz 1.0 --sine-phase-deg 0

注意：请确保接收端（scripts/test.py）的 --ros2-cmd-unit 与本发布器的 --unit 保持一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Tuple


def parse_pairs(pairs_str: str) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    if not pairs_str:
        return pairs
    for seg in pairs_str.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if ":" not in seg:
            raise ValueError(f"无法解析对 '{seg}'，应为 name:value")
        name, val = seg.split(":", 1)
        name = name.strip()
        try:
            v = float(val)
        except Exception:
            raise ValueError(f"角度值无法解析: '{val}' (pair '{seg}')")
        pairs.append((name, v))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="ROS2 关节命令发布器：发布 [[name, angle], ...] 的 JSON 字符串")
    parser.add_argument("--topic", type=str, default="/joint_cmd_array", help="发布话题名 (std_msgs/String)")
    parser.add_argument("--unit", type=str, choices=["rad", "deg"], default="rad", help="角度单位")
    parser.add_argument("--pairs", type=str, default="", help="关节命令对，格式 name:angle, 用逗号分隔，例如 l_elbow_joint:30,r_wrist_joint:-15")
    parser.add_argument("--rate", type=float, default=500.0, help="循环发布频率 Hz（若未指定 --oneshot）")
    parser.add_argument("--sine-amp", type=float, default=0.0, help="正弦摆幅值，单位与 --unit 一致（0 关闭正弦）")
    parser.add_argument("--sine-freq-hz", type=float, default=1.0, help="正弦摆频率 Hz（仅当摆幅>0 时生效）")
    parser.add_argument("--sine-phase-deg", type=float, default=0.0, help="初始相位（度），仅当摆幅>0 时生效")
    parser.add_argument("--oneshot", action="store_true", help="只发布一次后退出")
    args = parser.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String as RosString
    except Exception as e:
        print(f"[ERROR] 导入 rclpy 失败：{e}\n请先 source ROS2 环境。")
        sys.exit(1)

    try:
        pairs = parse_pairs(args.pairs)
        if not pairs:
            print("[ERROR] 未提供 --pairs，至少提供一个 name:angle")
            sys.exit(2)
    except Exception as e:
        print(f"[ERROR] 解析 --pairs 失败：{e}")
        sys.exit(2)

    # 基准角（围绕其做正弦），不进行单位转换，由接收端与本端协商单位
    base_pairs = [(name, float(val)) for name, val in pairs]
    use_sine = float(args.sine_amp) > 0.0 and float(args.sine_freq_hz) > 0.0
    phase_rad = float(args.sine_phase_deg) * 3.141592653589793 / 180.0

    class PubNode(Node):
        def __init__(self):
            super().__init__("joint_cmd_publisher")
            self.pub = self.create_publisher(RosString, args.topic, 10)
            self.msg = RosString()
            self._base_pairs = base_pairs
            self._use_sine = use_sine
            self._amp = float(args.sine_amp)
            self._freq = float(args.sine_freq_hz)
            self._phase = phase_rad
            self._t0 = self.get_clock().now().nanoseconds * 1e-9
            # 生成并写入一次 payload
            self._update_payload()
            self.oneshot = bool(args.oneshot)
            self.timer = None
            if self.oneshot:
                if self._use_sine:
                    self.get_logger().warn("oneshot 模式下忽略 --sine-* 参数，仅发布基准角。")
                # 立即发布一次
                self.pub.publish(self.msg)
                self.get_logger().info(f"已发布一次到 {args.topic}: {self.msg.data}")
                # 稍作延迟再退出，确保消息发送
                self.create_timer(0.1, self._shutdown)
            else:
                period = 1.0 / max(1e-3, float(args.rate))
                self.timer = self.create_timer(period, self._on_timer)
                sine_txt = (
                    f"，sine: amp={self._amp} {args.unit}, freq={self._freq} Hz, phase={args.sine_phase_deg} deg"
                    if self._use_sine else ""
                )
                self.get_logger().info(
                    f"开始以 {args.rate:.3f} Hz 循环发布到 {args.topic} (unit={args.unit}{sine_txt}): {self.msg.data}"
                )

        def _on_timer(self):
            # 周期性更新 payload（若启用正弦）并发布
            self._update_payload()
            self.pub.publish(self.msg)

        def _shutdown(self):
            # 触发一次后关闭
            try:
                rclpy.shutdown()
            except Exception:
                pass

        def _update_payload(self):
            # 根据是否启用正弦生成 payload
            if self._use_sine:
                now_s = self.get_clock().now().nanoseconds * 1e-9
                t = now_s - self._t0
                two_pi = 6.283185307179586
                val_list = []
                for name, base in self._base_pairs:
                    val = float(base) + self._amp * __import__('math').sin(two_pi * self._freq * t + self._phase)
                    val_list.append([name, float(val)])
                payload = json.dumps(val_list, ensure_ascii=False)
            else:
                payload = json.dumps([[n, float(v)] for n, v in self._base_pairs], ensure_ascii=False)
            self.msg.data = payload

    rclpy.init(args=None)
    node = PubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == "__main__":
    main()
