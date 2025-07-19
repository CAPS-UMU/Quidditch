# Error exporting nsnet2 with iree-turbine?
## Error:
```
[145/183] Translating NsNet2 using iree-turbine
FAILED: samples/nsnet2/nsnet2.mlirbc /home/hoppip/Quidditch/build/runtime/samples/nsnet2/nsnet2.mlirbc 
cd /home/hoppip/Quidditch/build/runtime/samples/nsnet2 && /home/hoppip/Quidditch/venv/bin/python3.11 /home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py /home/hoppip/Quidditch/build/runtime/samples/nsnet2/nsnet2.mlirbc --dtype=f64 --m=5 --n=6 --k=7
/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/cuda/__init__.py:619: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
Traceback (most recent call last):
  File "/home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py", line 105, in <module>
    exported = aot.export(with_frames(n_frames=args.frames))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/shark_turbine/aot/exporter.py", line 304, in export
    cm = TransformedModule(context=context, import_to="import")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/shark_turbine/aot/compiled_module.py", line 652, in __new__
    do_export(proc_def)
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/shark_turbine/aot/compiled_module.py", line 649, in do_export
    trace.trace_py_func(invoke_with_self)
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/shark_turbine/aot/support/procedural/tracer.py", line 122, in trace_py_func
    return_py_value = _unproxy(py_f(*self.proxy_posargs, **self.proxy_kwargs))
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/shark_turbine/aot/compiled_module.py", line 630, in invoke_with_self
    return proc_def.callable(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py", line 94, in main
    y, out1, out2 = aot.jittable(model.forward)(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/shark_turbine/aot/support/procedural/base.py", line 135, in __call__
    return current_ir_trace().handle_call(self, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/shark_turbine/aot/support/procedural/tracer.py", line 138, in handle_call
    return target.resolve_call(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/shark_turbine/aot/builtins/jittable.py", line 248, in resolve_call
    fx_importer.import_stateless_graph(gm.graph, func_name=self.function_name)
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/iree/compiler/extras/fx_importer.py", line 940, in import_stateless_graph
    node_importer.import_nodes(
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/iree/compiler/extras/fx_importer.py", line 1453, in import_nodes
    self._import_torch_op_overload(loc, node)
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/iree/compiler/extras/fx_importer.py", line 1668, in _import_torch_op_overload
    self._import_argument(loc, node.args[i], parameter.type)
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/iree/compiler/extras/fx_importer.py", line 1772, in _import_argument
    self.bind_node_value(arg, self._import_literal(obj))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hoppip/Quidditch/venv/lib/python3.11/site-packages/iree/compiler/extras/fx_importer.py", line 1835, in _import_literal
    user_value = self.fx_importer._hooks.resolve_literal(self, py_value, info)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: _Hooks.resolve_literal() takes 3 positional arguments but 4 were given
ninja: build stopped: subcommand failed.

```
## Hacky solution:
Change `user_value = self.fx_importer._hooks.resolve_literal(self, py_value, info)` to `user_value = self.fx_importer._hooks.resolve_literal(self, py_value)`. 

So from inside the build directory, do
```
cp ../fx_importer.py ../venv/lib/python3.11/site-packages/iree/compiler/extras/fx_importer.py
```