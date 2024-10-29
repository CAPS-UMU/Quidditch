# this script tries to compile some pomelo's MLIR with tiling scheme $1
# [hoppip@inf-205-141 zigzag_tiling]$ sh run.sh /home/hoppip/Quidditch/zigzag_tiling/zigzag-tile-scheme.json
echo $1

cd /home/hoppip/Quidditch/build/runtime/samples/pomelo;
/home/hoppip/Quidditch/build/codegen/iree-configuration/iree/tools/iree-compile \
--mlir-pretty-debuginfo \
--mlir-print-ir-after-all \
--iree-codegen-llvm-verbose-debug-info \
--iree-quidditch-zigzag-tiling-scheme=$1 \
--iree-quidditch-output-tiled=true \
--iree-vm-bytecode-module-strip-source-map=true \
--iree-vm-emit-polyglot-zip=false \
--iree-input-type=auto \
--iree-input-demote-f64-to-f32=0 \
--iree-hal-target-backends=quidditch \
--iree-quidditch-static-library-output-path=/home/hoppip/Quidditch/build/runtime/samples/pomelo/pamplemousse/pamplemousse.o \
--iree-quidditch-xdsl-opt-path=/home/hoppip/Quidditch/venv/bin/xdsl-opt \
--iree-quidditch-toolchain-root=/home/hoppip/Quidditch/toolchain \
--iree-quidditch-assert-compiled=true \
--output-format=vm-c \
--iree-vm-target-index-bits=32 \
/home/hoppip/Quidditch/runtime/samples/pomelo/pamplemousse.mlir \
-o /home/hoppip/Quidditch/build/runtime/samples/pomelo/pamplemousse/pamplemousse_module.h 2> /home/hoppip/Quidditch/zigzag_tiling/run_output.mlir
cd /home/hoppip/Quidditch/zigzag_tiling