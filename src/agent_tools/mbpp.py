import ast
import os
import platform
import subprocess
import sys

from src.agent_tools.base import BaseToolsAdapter, tool_method

TOOL_TIMEOUT = 10
MEM_LIMIT_BYTES = 256 * 1024 * 1024  # 256 MB

BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "pickle",
    "shelve",
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    "resource",
    "importlib",
}
BLOCKED_CALLS = {"open", "exec", "eval", "__import__", "compile", "input", "breakpoint"}


def _check_safety(code: str) -> str | None:
    """Return an error string if the code contains dangerous patterns, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.violation = None

        def visit_Import(self, node):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_MODULES:
                    self.violation = f"blocked import: '{alias.name}'"
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module:
                root = node.module.split(".")[0]
                if root in BLOCKED_MODULES:
                    self.violation = f"blocked import: '{node.module}'"
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                self.violation = f"blocked call: '{node.func.id}()'"
            self.generic_visit(node)

    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.violation


def _resource_limits():
    """Set CPU and memory limits in the child process (Linux only)."""
    if platform.system() != "Linux":
        return
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def _run(code: str, timeout: int = TOOL_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        timeout=timeout,
        capture_output=True,
        text=True,
        preexec_fn=_resource_limits,
    )


class MBPPTools(BaseToolsAdapter):
    def __init__(self, env=None):
        assert os.environ.get("ALLOW_CODE_EXECUTION"), (
            "MBPPTools executes arbitrary LLM-generated code in a subprocess. "
            "This is a potential code injection vector. "
            "We strongly recommend running inside a sandbox (e.g. Docker, gVisor, or nsjail). "
            "Set ALLOW_CODE_EXECUTION=1 to confirm you accept this risk."
        )
        self.env = env or {"code": "", "test_list": []}

    @tool_method
    def write_code(self, code: str) -> str:
        """Write or replace the current Python code in the coding environment.

        This tool stores the provided code so it can later be inspected,
        executed, or tested with other tools. Code is sanitized before being
        saved: Markdown code fences are stripped, and unsafe imports or calls
        are rejected.

        Args:
            code: Python source code to store in the environment.

        Returns:
            A confirmation string if the code was stored successfully,
            or a rejection message if the code violates safety checks.
        """
        code = code.replace("```python", "").replace("```", "").strip()
        violation = _check_safety(code)
        if violation:
            return f"[rejected — {violation}]"
        self.env["code"] = code
        return "Code written successfully."

    @tool_method
    def read_code(self) -> str:
        """Read the current Python code stored in the coding environment.

        Returns:
            The full stored code as a string, or "(empty)" if no code
            has been written yet.
        """
        code = self.env.get("code", "")
        return code if code else "(empty)"

    @tool_method
    def run_code(self) -> str:
        """Execute the currently stored Python code and return its output.

        The stored code is run in a restricted subprocess with time and memory
        limits. Both standard output and standard error are captured. If the
        program exits with a non-zero status, the exit code is included in the
        returned message.

        Returns:
            A string containing stdout, optional stderr, and optional exit code.
            Returns an error-style message if no code is stored, execution times
            out, or another execution error occurs.
        """
        code = self.env.get("code", "")
        if not code:
            return "[coding environment is empty — use write_code first]"
        try:
            result = _run(code)
            out = result.stdout
            if result.stderr:
                out += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                out += f"\n[exit code: {result.returncode}]"
            return out or "(no output)"
        except subprocess.TimeoutExpired:
            return f"[timeout after {TOOL_TIMEOUT}s]"
        except Exception as e:
            return f"[error: {e}]"

    @tool_method
    def check_tests(self) -> str:
        """Run all stored test assertions against the current code.

        Each test is executed together with the stored code in an isolated,
        restricted subprocess. The result reports whether each test passed,
        failed, timed out, or raised an execution error.

        Returns:
            A summary string containing the number of passed tests and
            detailed per-test results. Returns an error-style message if
            no code is stored or no tests are available.
        """
        code = self.env.get("code", "")
        if not code:
            return "[coding environment is empty — use write_code first]"
        test_list = self.env.get("test_list", [])
        if not test_list:
            return "[no test cases available]"

        results = []
        for i, test in enumerate(test_list):
            try:
                result = _run(code + "\n" + test)
                if result.returncode == 0:
                    results.append(f"Test {i + 1}: PASS  — {test}")
                else:
                    err = (result.stderr or result.stdout or "").strip()
                    results.append(f"Test {i + 1}: FAIL — {test}\n  {err}")
            except subprocess.TimeoutExpired:
                results.append(f"Test {i + 1}: TIMEOUT — {test}")
            except Exception as e:
                results.append(f"Test {i + 1}: ERROR — {e}")

        passed = sum(1 for r in results if ": PASS" in r)
        return f"Passed {passed}/{len(test_list)}\n\n" + "\n".join(results)

    @tool_method
    def write_tests(self, tests: list[str]) -> str:
        """Append new test assertions to the current test list.

        Tests should be valid Python assertions that can be executed together
        with the stored code, for example function calls followed by expected
        outputs.

        Args:
            tests: A list of test assertion strings, such as
                ["assert add(2, 3) == 5", "assert add(0, 0) == 0"].

        Returns:
            A confirmation string indicating how many tests were added.
        """
        self.env["test_list"].extend(tests)
        return f"{len(tests)} test(s) written successfully."
