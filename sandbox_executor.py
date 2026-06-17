import subprocess
import sys
import tempfile
import traceback

class SandboxExecutor:
    def __init__(self, timeout=5.0):
        self.timeout = timeout

    def run_python_code(self, code: str):
        """
        Executes Python code in a temporary file and returns stdout, stderr, and success status.
        """
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=True) as temp_file:
            temp_file.write(code)
            temp_file.flush()
            
            try:
                result = subprocess.run(
                    [sys.executable, temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Execution timed out after {self.timeout} seconds.",
                    "returncode": -1
                }
            except Exception as e:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Sandbox execution error: {str(e)}\n{traceback.format_exc()}",
                    "returncode": -1
                }

    def run_shell_command(self, cmd: str):
        """
        Executes a shell command securely and returns result stats.
        """
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout} seconds.",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Shell sandbox error: {str(e)}",
                "returncode": -1
            }

if __name__ == "__main__":
    # Quick self-test
    sandbox = SandboxExecutor()
    res = sandbox.run_python_code("print('Hello from sandbox!')")
    print("Test Output:", res)
