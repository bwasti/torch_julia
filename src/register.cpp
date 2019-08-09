#define JULIA_ENABLE_THREADING
#include <julia.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator_options.h>

#include <iostream>

namespace py = pybind11;
using namespace torch::jit;

at::Tensor from_julia_array(jl_array_t *x) {
  int ndims = jl_array_ndims(x);
  std::vector<int64_t> sizes;
  for (int i = 0; i < ndims; ++i) {
    sizes.emplace_back(jl_array_dim(x, i));
  }
  auto t = at::empty(sizes);
  float *p = (float *)jl_array_data(x);
  if (ndims == 1) {
    auto size0 = jl_array_dim(x, 0);
    for (int i = 0; i < size0; ++i) {
      t.data<float>()[i] = p[i];
    }
  } else if (ndims == 2) {
    auto size0 = jl_array_dim(x, 0);
    for (int i = 0; i < size0; ++i) {
      for (int j = 0; j < jl_array_dim(x, 1); ++j) {
        t.data<float>()[i * size0 + j] = p[j * size0 + i];
      }
    }
  } else {
    std::cerr << "unhandled dim greater than 2\n";
    exit(1);
  }
  return t;
}

jl_array_t *to_julia_array(at::Tensor &t) {
  t.contiguous();
  jl_value_t *array_type =
      jl_apply_array_type((jl_value_t *)jl_float32_type, t.dim());
  jl_array_t *x;
  switch (t.dim()) {
  case 1:
    x = jl_alloc_array_1d(array_type, t.size(0));
    break;
  case 2:
    x = jl_alloc_array_2d(array_type, t.size(0), t.size(1));
    break;
  default:
    std::cerr << "unhandled dim greater than 2\n";
    exit(1);
  }
  float *p = (float *)jl_array_data(x);
  if (t.dim() == 1) {
    auto size0 = t.size(0);
    for (int i = 0; i < size0; ++i) {
      p[i] = t.data<float>()[i];
    }
  }
  if (t.dim() == 2) {
    auto size0 = t.size(0);
    for (int i = 0; i < size0; ++i) {
      for (int j = 0; j < t.size(1); ++j) {
        p[j * size0 + i] = t.data<float>()[i * size0 + j];
      }
    }
  }
  return x;
}

void load_module(std::string fn, std::string module_name = "") {
  JL_TRY {
    if (module_name.size()) {
      auto module = jl_new_module(jl_symbol(module_name.c_str()));
      jl_load(module, fn.c_str());
    } else {
      jl_load(jl_main_module, fn.c_str());
    }
  }
  JL_CATCH {
    jl_value_t *errs = jl_stderr_obj();
    volatile int shown_err = 0;
    jl_printf(JL_STDERR, "error during bootstrap:\n");
    JL_TRY {
      if (errs) {
        jl_value_t *showf = jl_get_function(jl_base_module, "show");
        if (showf != NULL) {
          jl_call2(showf, errs, jl_current_exception());
          jl_printf(JL_STDERR, "\n");
          shown_err = 1;
        }
      }
    }
    JL_CATCH {}
    if (!shown_err) {
      jl_static_show(JL_STDERR, jl_current_exception());
      jl_printf(JL_STDERR, "\n");
    }
    jlbacktrace();
    jl_printf(JL_STDERR, "\n");
    exit(1);
  }
}

void import_function(std::string func_name, size_t num_args) {
  jl_value_t *func =
      jl_get_global(jl_main_module, jl_symbol(func_name.c_str()));
  std::vector<Argument> args;
  for (auto i = 0; i < num_args; ++i) {
    args.emplace_back(std::string("arg_") + std::to_string(i),
                      getTypePtr<at::Tensor>());
  }
  // NB: We assume all julia ops are pure
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(AliasAnalysisKind::PURE);
  auto torch_operator = Operator(
      FunctionSchema("julia::" + func_name, "", args, {}, false, true),
      [func, num_args](Stack &stack) {
        // RECORD_FUNCTION("julia::" + fn, std::vector<c10::IValue>());
        jl_value_t **jl_args;
        JL_GC_PUSHARGS(jl_args, num_args);
        for (auto i = 0; i < num_args; ++i) {
          auto tensor_ = stack.back().toTensor();
          stack.pop_back();
          auto x = to_julia_array(tensor_);
          jl_args[num_args - i - 1] = (jl_value_t *)x;
        }

        auto y = jl_call(func, jl_args, num_args);
        JL_GC_POP();

        if (jl_exception_occurred()) {
          printf("func is bad: %s \n", jl_typeof_str(jl_exception_occurred()));
          exit(1);
        }
        auto tensor = from_julia_array((jl_array_t *)y);
        auto var = torch::autograd::make_variable(tensor);
        stack.push_back(IValue(var));
        return 0;
      }, options);
  RegisterOperators torch_register_ops(std::vector<Operator>{torch_operator});
}

void load(std::string fn) {
  load_module(fn);
  jl_array_t *reg_funcs =
      (jl_array_t *)jl_get_global(jl_main_module, jl_symbol("torch_funcs"));
  for (auto i = 0; i < jl_array_len(reg_funcs); ++i) {
    jl_value_t *s = jl_array_ptr_ref(reg_funcs, i);
    jl_sym_t *name = (jl_sym_t *)jl_get_nth_field(s, 0);
    jl_array_t *arities = (jl_array_t *)jl_get_nth_field(s, 1);
    auto name_str = std::string(jl_symbol_name(name));
    for (auto j = 0; j < jl_array_len(arities); ++j) {
      auto arity = ((int64_t *)jl_array_data(arities))[j];
      import_function(name_str, arity);
    }
  }
}

PYBIND11_MODULE(torch_julia, m) {
  jl_init();
  if (jl_exception_occurred()) {
    printf("julia is bad: %s \n", jl_typeof_str(jl_exception_occurred()));
    exit(1);
  }

  m.def("load", &load, "Load the julia file");

  // This is a hack to get jl_atexit_hook working
  auto cleanup_callback = []() { jl_atexit_hook(0); };
  m.add_object("_cleanup", py::capsule(cleanup_callback));
}
