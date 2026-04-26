import argparse
import sys
from pathlib import Path

"""
SPAD 命令行总入口模块。

模块目的：
- 在 train/test 子命令之间做分发，复用统一入口。

主要导出内容：
- main: 解析一级命令并转发到训练或测试脚本。
"""

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scipts import test as test_script
from scipts import train as train_script


def main() -> None:
	"""解析一级命令并分发到训练或测试流程。

	Returns:
		None。

	Raises:
		ValueError: 收到不支持的 command。
	"""
	parser = argparse.ArgumentParser(description="SPAD pipeline entrypoint")
	parser.add_argument("command", choices=["train", "test"], help="Run training or testing")
	args, remaining = parser.parse_known_args()

	if args.command == "train":
		train_script.main(remaining)
		return

	if args.command == "test":
		test_script.main(remaining)
		return

	raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
	main()
