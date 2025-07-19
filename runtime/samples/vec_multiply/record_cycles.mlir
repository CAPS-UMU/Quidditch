// tried to put record_cycles in its own module, but now I get a
// `outer module does not contain a vm.module op` error
builtin.module @record_cycles {
      "func.func"() <{function_type = (i32) 
  -> (), sym_name = "record_cycles", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()
}
