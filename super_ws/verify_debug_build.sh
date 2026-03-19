#!/usr/bin/env bash
# 确认 super_planner 是 Debug 构建且当前运行的 fsm_node 用的是这份二进制
set -e
WS=/mnt/data/super_ws
BUILD="$WS/build/super_planner"
INSTALL_EXE="$WS/install/super_planner/lib/super_planner/fsm_node"

echo "=== 1. 检查 CMake 配置（是否 Debug、是否 -O0）==="
if [[ -f "$BUILD/CMakeCache.txt" ]]; then
  grep -E "CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS[^_]" "$BUILD/CMakeCache.txt" || true
  if grep -q "CMAKE_BUILD_TYPE:STRING=Debug" "$BUILD/CMakeCache.txt" 2>/dev/null; then
    echo "[OK] CMAKE_BUILD_TYPE=Debug"
  else
    echo "[!!] 不是 Debug 构建，断点可能断不住。请执行："
    echo "     rm -rf $BUILD $WS/install/super_planner"
    echo "     cd $WS && colcon build --packages-select super_planner --cmake-args -DCMAKE_BUILD_TYPE=Debug"
  fi
else
  echo "[!!] 未找到 $BUILD，请先 colcon build。"
fi

echo ""
echo "=== 2. 当前正在运行的 fsm_node 进程及其可执行文件 ==="
PIDS=$(pgrep -f "fsm_node" || true)
if [[ -z "$PIDS" ]]; then
  echo "[!!] 没有正在运行的 fsm_node。请先 ros2 launch 启动节点后再附加 GDB。"
else
  for pid in $PIDS; do
    exe=$(readlink -f /proc/$pid/exe 2>/dev/null || echo "?")
    echo "  PID $pid -> $exe"
    if [[ "$exe" != "$(readlink -f $INSTALL_EXE 2>/dev/null)" ]]; then
      echo "    [!!] 与 install 路径不一致，请确认附加的是此 PID：$pid"
    fi
  done
fi

echo ""
echo "=== 3. install 里的 fsm_node 是否存在、是否含调试信息 ==="
if [[ -f "$INSTALL_EXE" ]]; then
  file "$INSTALL_EXE"
else
  echo "[!!] 不存在: $INSTALL_EXE"
fi
