# this script tries to compile some calabaza's MLIR with tiling scheme $1
# sh boo.sh /home/hoppip/Quidditch/zigzag_tiling/zigzag-tile-scheme.json
# sh boo.sh /home/hoppip/Quidditch/zigzag_tiling/zigzag-tile-scheme.json
# sh boo.sh strawberry.json
# sh boo.sh hoodle.json
echo $1

cd /home/hoppip/Quidditch/build/runtime/samples/calabaza;
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
--iree-quidditch-static-library-output-path=/home/hoppip/Quidditch/build/runtime/samples/calabaza/pumpkin/pumpkin.o \
--iree-quidditch-xdsl-opt-path=/home/hoppip/Quidditch/venv/bin/xdsl-opt \
--iree-quidditch-toolchain-root=/home/hoppip/Quidditch/toolchain \
--iree-quidditch-assert-compiled=true \
--output-format=vm-c \
--iree-vm-target-index-bits=32 \
/home/hoppip/Quidditch/runtime/samples/calabaza/pumpkin.mlir \
-o /home/hoppip/Quidditch/build/runtime/samples/calabaza/pumpkin/pumpkin_module.h 2> /home/hoppip/Quidditch/zigzag_tiling/boo_output.mlir
cd /home/hoppip/Quidditch/zigzag_tiling