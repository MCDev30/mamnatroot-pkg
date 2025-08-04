import numpy as np
import time
from typing import Callable, List, Tuple, Union

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class MamNatRootSolver:
    """
    Implements the hybrid MamNatRoot method for root finding.
    This class provides only static methods and does not need to be instantiated.
    """

    @staticmethod
    def find_all_roots(
        func: Callable[[float], float],
        interval: Union[List[float], Tuple[float, float]],
        depth: int = 12,
        verbose: bool = False,
        visualize: bool = False,
        getRuntime: bool = False
    ) -> Union[List[float], Tuple[List[float], float]]:
        """
        Find all roots of a function using the MamNatRoot method.

        Parameters
        ----------
        func : Callable[[float], float]
            The function to analyze.
        interval : list or tuple of float
            The search interval, as a list or tuple [a, b].
        depth : int, optional
            The depth of subdivision to isolate roots (default is 12).
        verbose : bool, optional
            If True, prints calculation steps to the console (default is False).
        visualize : bool, optional
            If True, displays a plot of the results (requires matplotlib, default is False).
        getRuntime : bool, optional
            If True, returns a tuple (roots, execution_time) (default is False).

        Returns
        -------
        list of float or tuple (list of float, float)
            A list of roots, or a tuple (roots, execution_time) if getRuntime is True.

        Raises
        ------
        ValueError
            If the interval is not a list or tuple of two numbers [a, b] with a < b.
        """
        start_time = time.perf_counter()

        if len(interval) != 2 or interval[0] >= interval[1]:
            raise ValueError("Interval must be a list or tuple of two numbers [a, b] with a < b.")
        a, b = interval

        if verbose:
            print("--- START MamNatRoot SOLVER ---")
            print(f"Interval: [{a}, {b}], Depth: {depth}")

        # Step 1: Isolate intervals containing roots
        if verbose:
            print("\n1. Root isolation phase...")
        root_intervals = MamNatRootSolver._isolate_roots_intervals(func, a, b, 0, depth)

        if not root_intervals:
            if verbose:
                print("No interval with sign change was found.")
            return ([], 0.0) if getRuntime else []

        if verbose:
            print(f"  {len(root_intervals)} interval(s) found.")

        # Step 2: Local approximation
        if verbose:
            print("\n2. Local approximation phase...")
        found_roots = []
        for iv in root_intervals:
            if verbose:
                print(f"  Processing interval [{iv[0]:.4f}, {iv[1]:.4f}]...")
            root = MamNatRootSolver._approximate_root_in_interval(func, iv, verbose=verbose)
            found_roots.append(root)
            if verbose:
                print(f"    -> Approximate root found at x = {root:.6f}")

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if verbose:
            print("\n--- END MamNatRoot SOLVER ---")
            print(f"Total computation time: {execution_time:.6f} seconds")

        # Step 3: Optional visualization
        if visualize:
            if not MATPLOTLIB_AVAILABLE:
                print("\nWarning: `matplotlib` is not installed. Visualization not possible.")
                print("Install it with: pip install mamnatroot[visualize]")
            else:
                if verbose:
                    print("Generating plot...")
                x_vals = np.linspace(a, b, 500)
                y_vals = func(x_vals)
                plt.plot(x_vals, y_vals, label='f(x)', color='blue')
                plt.axhline(0, color='gray', linestyle='--')
                plt.plot(found_roots, func(np.array(found_roots)), 'rX', markersize=5)
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.legend()
                plt.grid(True)
                plt.show()

        # Step 4: Return result according to options
        if getRuntime:
            return (found_roots, execution_time)
        else:
            return found_roots

    @staticmethod
    def _isolate_roots_intervals(
        func: Callable[[float], float],
        a: float,
        b: float,
        current_depth: int,
        max_depth: int
    ) -> List[Tuple[float, float]]:
        """
        Recursively find refined subintervals containing a root.

        Parameters
        ----------
        func : Callable[[float], float]
            The function to analyze.
        a : float
            Left endpoint of the interval.
        b : float
            Right endpoint of the interval.
        current_depth : int
            Current recursion depth.
        max_depth : int
            Maximum recursion depth.

        Returns
        -------
        list of tuple (float, float)
            List of intervals [a, b] where a sign change occurs.
        """
        # Base case: reached maximum recursion depth
        if current_depth == max_depth:
            # If the function changes sign at the endpoints, keep this interval
            if func(a) * func(b) < 0:
                return [(a, b)]
            return []

        # Recursive step: split the interval in two and explore each half
        mid = (a + b) / 2
        left_intervals = MamNatRootSolver._isolate_roots_intervals(func, a, mid, current_depth + 1, max_depth)
        right_intervals = MamNatRootSolver._isolate_roots_intervals(func, mid, b, current_depth + 1, max_depth)

        # Return the combined list of intervals found on the left and right
        return left_intervals + right_intervals

    @staticmethod
    def _approximate_root_in_interval(
        func: Callable[[float], float],
        interval: Tuple[float, float],
        verbose: bool = False
    ) -> float:
        """
        Find an approximation of the root in a given interval
        using quadratic interpolation (parabola).

        Parameters
        ----------
        func : Callable[[float], float]
            The function to analyze.
        interval : tuple (float, float)
            The interval [c, d] in which to approximate the root.
        verbose : bool, optional
            If True, prints warnings in case of fallback (default is False).

        Returns
        -------
        float
            An approximate root within the interval.
        """
        c, d = interval

        # 1. Generate 3 points (endpoints and midpoint) to define the parabola
        x_points = np.array([c, (c + d) / 2, d])
        y_points = func(x_points)

        # 2. Determine the coefficients of P(x) = alpha*x^2 + beta*x + gamma
        A = np.vstack([x_points**2, x_points, np.ones(3)]).T
        try:
            alpha, beta, gamma = np.linalg.solve(A, y_points)
        except np.linalg.LinAlgError:
            # In case of calculation error (colinear points...), return the midpoint
            return (c + d) / 2

        # 3. Handle the case where the approximation is a line (alpha ~ 0)
        if abs(alpha) < 1e-12:
            return -gamma / beta if abs(beta) > 1e-12 else (c + d) / 2

        # 4. Compute the zeros of the parabola using the quadratic formula
        discriminant = beta**2 - 4 * alpha * gamma

        # If the discriminant is negative, the parabola does not cross the x-axis.
        # Use a simple approximation as a fallback.
        if discriminant < 0:
            if verbose:
                print(f"Warning: Negative discriminant. Simple approximation used.")
            return (c + d) / 2

        # Compute the two possible roots
        sqrt_discriminant = np.sqrt(discriminant)
        root1 = (-beta - sqrt_discriminant) / (2 * alpha)
        root2 = (-beta + sqrt_discriminant) / (2 * alpha)

        # 5. Choose the root that lies within our confidence interval [c, d]
        if c <= root1 <= d:
            return root1
        elif c <= root2 <= d:
            return root2
        else:
            # If neither root is in the interval (rare), use the fallback as well.
            if verbose:
                print(f"Warning: Parabola root out of interval. Simple approximation used.")
            return (c + d) / 2