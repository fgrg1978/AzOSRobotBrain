# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Allow running: python -m tools.robot_cli ..."""

from tools.robot_cli import main
import sys

sys.exit(main())
