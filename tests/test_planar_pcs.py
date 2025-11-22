import dill
import jax

jax.config.update("jax_enable_x64", True)  # double precision
from jax import Array
from jax import numpy as jnp
import jsrm
from functools import partial
from numpy.testing import assert_allclose
from pathlib import Path
import sympy as sp

from jsrm.systems import planar_pcs, euler_lagrangian
from jsrm.utils.tolerance import Tolerance


def constant_strain_inverse_kinematics_fn(params, xi_eq, chi, s) -> Array:
    # split the chi vector into x, y, and th0
    px, py, th = chi
    th0 = params["th0"].item()
    print("th0 = ", th0)
    # xi = (th - th0) / (2 * s) * jnp.array([
    #     2.0, 
    #     (jnp.sin(th0)*px+jnp.cos(th0)*py) - (jnp.cos(th0)*px-jnp.sin(th0)*py)*jnp.sin(th-th0)/(jnp.cos(th-th0)-1), 
    #     -(jnp.cos(th0)*px-jnp.sin(th0)*py) - (jnp.sin(th0)*px+jnp.cos(th0)*py)*jnp.sin(th-th0)/(jnp.cos(th-th0)-1)
    # ])
    xi = (th - th0) / (2 * s) * jnp.array([
        2.0, 
        (-jnp.sin(th0)*px+jnp.cos(th0)*py) - (jnp.cos(th0)*px+jnp.sin(th0)*py)*jnp.sin(th-th0)/(jnp.cos(th-th0)-1), 
        -(jnp.cos(th0)*px+jnp.sin(th0)*py) - (-jnp.sin(th0)*px+jnp.cos(th0)*py)*jnp.sin(th-th0)/(jnp.cos(th-th0)-1)
    ])
    q = xi - xi_eq
    return q

def test_planar_cc():
    sym_exp_filepath = (
        Path(jsrm.__file__).parent / "symbolic_expressions" / "planar_pcs_ns-1.dill"
    )
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    xi_syms = sym_exps["state_syms"]["xi"]
    xi_d_syms = sym_exps["state_syms"]["xi_d"]

    num_segments = len(xi_syms) // 3
    shear_indices = [3 * i + 1 for i in range(num_segments)]
    axial_indices = [3 * i + 2 for i in range(num_segments)]

    substitutions = {sym_exps["params_syms"]["th0"]: 0}
    for idx in shear_indices:
        substitutions[xi_syms[idx]] = 0
        substitutions[xi_d_syms[idx]] = 0
    for idx in axial_indices:
        substitutions[xi_syms[idx]] = 1
        substitutions[xi_d_syms[idx]] = 1

    forbidden_syms = set(substitutions.keys())
    bending_syms = [xi_syms[3 * i] for i in range(num_segments)]

    def remove_rows_cols(mat: sp.Matrix, remove_idxs):
        mat_mutable = sp.Matrix(mat)
        keep_rows = [i for i in range(mat_mutable.rows) if i not in remove_idxs]
        keep_cols = [i for i in range(mat_mutable.cols) if i not in remove_idxs]
        return mat_mutable.extract(keep_rows, keep_cols)

    def remove_cols(mat: sp.Matrix, remove_idxs):
        mat_mutable = sp.Matrix(mat)
        keep_cols = [i for i in range(mat_mutable.cols) if i not in remove_idxs]
        keep_rows = list(range(mat_mutable.rows))
        return mat_mutable.extract(keep_rows, keep_cols)

    def remove_rows(mat: sp.Matrix, remove_idxs):
        mat_mutable = sp.Matrix(mat)
        keep_rows = [i for i in range(mat_mutable.rows) if i not in remove_idxs]
        return mat_mutable.extract(keep_rows, [0])

    simplified_exps = {}
    expected_dim = len(xi_syms) - len(shear_indices) - len(axial_indices)
    expected_j_cols = len(xi_syms) // 3  # one bending DOF per segment

    for exp_key, exp_val in sym_exps["exps"].items():
        def simplify_and_reduce(expr: sp.Expr) -> sp.Expr:
            simplified_expr = sp.simplify(expr.subs(substitutions))
            if exp_key in {"B", "C"}:
                simplified_expr = remove_rows_cols(simplified_expr, shear_indices + axial_indices)
                assert simplified_expr.shape == (expected_dim, expected_dim)
            elif exp_key == "G":
                simplified_expr = remove_rows(simplified_expr, shear_indices + axial_indices)
                assert simplified_expr.shape == (expected_dim, 1)
            elif exp_key in {"J_sms", "J_d_sms", "Jee", "Jee_d"}:
                simplified_expr = remove_cols(simplified_expr, shear_indices + axial_indices)
                assert simplified_expr.shape == (simplified_expr.rows, expected_j_cols)
            elif exp_key == "J_tend_sms":
                simplified_expr = remove_cols(simplified_expr, shear_indices + axial_indices)
                assert simplified_expr.shape == (simplified_expr.rows, expected_j_cols)
            return simplified_expr

        if isinstance(exp_val, list):
            simplified_list = []
            for idx, exp_item in enumerate(exp_val):
                simplified_item = simplify_and_reduce(exp_item)
                simplified_list.append(simplified_item)
                print(f"{exp_key}[{idx}] =\n{simplified_item}")
                assert forbidden_syms.isdisjoint(simplified_item.free_symbols)
            simplified_exps[exp_key] = simplified_list
        else:
            simplified_item = simplify_and_reduce(exp_val)
            simplified_exps[exp_key] = simplified_item
            print(f"{exp_key} =\n{simplified_item}")
            assert forbidden_syms.isdisjoint(simplified_item.free_symbols)

    def limit_bending_strain(expr: sp.Expr) -> sp.Expr:
        limited_expr = expr
        for bending_sym in bending_syms:
            limited_expr = sp.limit(limited_expr, bending_sym, 0)
        return sp.simplify(limited_expr)

    def assert_no_bending_syms(expr: sp.Expr):
        assert set(bending_syms).isdisjoint(getattr(expr, "free_symbols", set()))

    limited_exps = {}
    for exp_key, exp_val in simplified_exps.items():
        if isinstance(exp_val, list):
            limited_list = []
            for idx, exp_item in enumerate(exp_val):
                limited_item = (
                    exp_item.applyfunc(limit_bending_strain)
                    if isinstance(exp_item, sp.MatrixBase)
                    else limit_bending_strain(exp_item)
                )
                limited_list.append(limited_item)
                print(f"{exp_key}_limit[{idx}] =\n{limited_item}")
                assert_no_bending_syms(limited_item)
            limited_exps[exp_key] = limited_list
        else:
            limited_item = (
                exp_val.applyfunc(limit_bending_strain)
                if isinstance(exp_val, sp.MatrixBase)
                else limit_bending_strain(exp_val)
            )
            limited_exps[exp_key] = limited_item
            print(f"{exp_key}_limit =\n{limited_item}")
            assert_no_bending_syms(limited_item)

def test_planar_cs():
    sym_exp_filepath = (
        Path(jsrm.__file__).parent / "symbolic_expressions" / "planar_pcs_ns-1.dill"
    )
    params = {
        "th0": jnp.array(0.0),  # initial orientation angle [rad]
        "l": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "rho": 1000 * jnp.ones((1,)),
        "g": jnp.array([0.0, -9.81]),
        "E": 1e8 * jnp.ones((1,)),  # Elastic modulus [Pa]
        "G": 1e7 * jnp.ones((1,)),  # Shear modulus [Pa]
    }
    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((3,), dtype=bool)

    xi_eq = jnp.array([0.0, 0.0, 1.0])
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
        planar_pcs.factory(sym_exp_filepath, strain_selector, xi_eq)
    )
    forward_dynamics_fn = partial(
        euler_lagrangian.forward_dynamics, dynamical_matrices_fn
    )
    nonlinear_state_space_fn = partial(
        euler_lagrangian.nonlinear_state_space, dynamical_matrices_fn
    )

    # test forward kinematics
    assert_allclose(
        forward_kinematics_fn(params, q=jnp.zeros((3,)), s=params["l"][0] / 2),
        jnp.array([0.0, params["l"][0] / 2, 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        forward_kinematics_fn(params, q=jnp.zeros((3,)), s=params["l"][0]),
        jnp.array([0.0, params["l"][0], 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        forward_kinematics_fn(params, q=jnp.array([0.0, 0.0, 1.0]), s=params["l"][0]),
        jnp.array([0.0, 2 * params["l"][0], 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        forward_kinematics_fn(params, q=jnp.array([0.0, 1.0, 0.0]), s=params["l"][0]),
        params["l"][0] * jnp.array([1.0, 1.0, 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )

    # test inverse kinematics
    params_ik = params.copy()
    ik_th0_ls = [-jnp.pi / 2, -jnp.pi / 4, 0.0, jnp.pi / 4, jnp.pi / 2]
    ik_q_ls = [
        jnp.array([0.1, 0.0, 0.0]),
        jnp.array([0.1, 0.0, 0.2]),
        jnp.array([0.1, 0.5, 0.1]),
        jnp.array([1.0, 0.5, 0.2]),
        jnp.array([-1.0, 0.0, 0.0]),
    ]
    for ik_th0 in ik_th0_ls:
        params_ik["th0"] = jnp.array(ik_th0)
        for q in ik_q_ls:
            s = params["l"][0]
            chi = forward_kinematics_fn(params_ik, q=q, s=s)
            q_ik = constant_strain_inverse_kinematics_fn(params_ik, xi_eq, chi, s)
            print("q = ", q, "q_ik = ", q_ik)
            assert_allclose(q, q_ik, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    # test dynamical matrices
    q, q_d = jnp.zeros((3,)), jnp.zeros((3,))
    B, C, G, K, D, A = dynamical_matrices_fn(params, q, q_d)
    assert_allclose(K, jnp.zeros((3,)))
    assert_allclose(
        A,
        jnp.eye(3),
    )

    q = jnp.array([jnp.pi / (2 * params["l"][0]), 0.0, 0.0])
    q_d = jnp.zeros((3,))
    B, C, G, K, D, alpha = dynamical_matrices_fn(params, q, q_d)

    print("B =\n", B)
    print("C =\n", C)
    print("G =\n", G)
    print("K =\n", K)
    print("D =\n", D)
    print("alpha =\n", alpha)


if __name__ == "__main__":
    test_planar_cc()
    exit()
    test_planar_cs()
