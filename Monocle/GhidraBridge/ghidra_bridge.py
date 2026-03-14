import concurrent
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm


class GhidraBridge():
    DEFAULT_MAXMEM = "2G"
    DEFAULT_THREADS = 2

    def __init__(self, ghidra_path=None, maxmem=None, threads=None):
        self.headless_path = self._resolve_ghidra_path(ghidra_path)
        self._maxmem_setting = maxmem
        self._threads_setting = threads

    @staticmethod
    def _get_total_ram_gb():
        try:
            return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def auto_maxmem(binary_path):
        """Calculate Java heap size based on binary size and available RAM."""
        binary_size_mb = Path(binary_path).stat().st_size / (1024 * 1024)

        if binary_size_mb < 10:
            heap_gb = 2
        elif binary_size_mb < 50:
            heap_gb = 4
        elif binary_size_mb < 200:
            heap_gb = 8
        else:
            heap_gb = 16

        total_ram = GhidraBridge._get_total_ram_gb()
        if total_ram:
            max_heap = int(total_ram / 4)
            heap_gb = min(heap_gb, max(2, max_heap))

        return f"{heap_gb}G"

    @staticmethod
    def auto_threads():
        """Calculate thread count based on available CPUs."""
        cpu_count = os.cpu_count() or 4
        return max(2, min(cpu_count // 2, 16))

    @staticmethod
    def _headless_binary_name():
        if platform.system() == "Windows":
            return "analyzeHeadless.bat"
        return "analyzeHeadless"

    @staticmethod
    def _resolve_java_home():
        java_home = os.environ.get("JAVA_HOME")
        if java_home and Path(java_home).is_dir():
            return java_home
        java_bin = shutil.which("java")
        if java_bin:
            real_path = Path(java_bin).resolve()
            candidate = real_path.parent.parent
            if (candidate / "bin" / "java").exists():
                return str(candidate)
        return None

    def _resolve_ghidra_path(self, ghidra_path=None):
        binary_name = self._headless_binary_name()

        if ghidra_path:
            p = Path(ghidra_path)
            if p.is_file() and p.exists():
                return str(p)
            candidate = p / "support" / binary_name
            if candidate.exists():
                return str(candidate)
            raise FileNotFoundError(f"Ghidra not found at: {ghidra_path}")

        env_home = os.environ.get("GHIDRA_HOME")
        if env_home:
            candidate = Path(env_home) / "support" / binary_name
            if candidate.exists():
                return str(candidate)

        on_path = shutil.which(binary_name)
        if on_path:
            return on_path

        return None

    def _execute_blocking_command(self, command_as_list):
        if command_as_list is None:
            return None
        str_command = [str(arg) for arg in command_as_list]

        env = os.environ.copy()
        if "JAVA_HOME" not in env:
            java_home = self._resolve_java_home()
            if java_home:
                env["JAVA_HOME"] = java_home

        result = subprocess.run(
            str_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        stdout_text = result.stdout.decode("utf-8", errors="replace").strip()
        stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
        if result.returncode != 0:
            error_detail = stderr_text or stdout_text or "(no output)"
            raise RuntimeError(
                f"Ghidra exited with code {result.returncode}:\n{error_detail}"
            )
        self.last_stdout = stdout_text
        self.last_stderr = stderr_text
        return result

    def generate_ghidra_decom_script(self, path_to_save_decoms_to, file_to_save_script_to):
        escaped_path = path_to_save_decoms_to.replace("\\", "\\\\")

        script = """//DecomScript.java
//@category Monocle
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import java.io.*;

public class DecomScript extends GhidraScript {
    @Override
    public void run() throws Exception {
        String outputDirectory = "<PATH>";

        DecompInterface decompiler = new DecompInterface();
        decompiler.openProgram(currentProgram);

        File dir = new File(outputDirectory);
        if (!dir.exists()) dir.mkdirs();

        FunctionIterator functions = currentProgram.getListing().getFunctions(true);
        while (functions.hasNext()) {
            Function function = functions.next();
            String functionName = function.getName();

            DecompileResults results = decompiler.decompileFunction(function, 0, monitor);
            if (results != null && results.getDecompiledFunction() != null) {
                String cCode = results.getDecompiledFunction().getC();

                String fileName = (currentProgram.getName() + "__" + functionName
                    + "__" + System.currentTimeMillis() + ".c")
                    .replaceAll("[^\\\\w\\\\-\\\\.]", "_");

                File outputFile = new File(dir, fileName);
                try (FileWriter writer = new FileWriter(outputFile)) {
                    writer.write(cCode);
                }
            }
        }

        decompiler.dispose();
    }
}
""".replace("<PATH>", escaped_path)

        with open(file_to_save_script_to, "w") as file:
            file.write(script)

    def _check_if_ghidra_project_exists(self, project_folder, project_name):

        project_folder_path = Path(project_folder, project_name + ".gpr")

        return project_folder_path.exists()

    def _resolve_launch_script(self):
        """Resolve path to Ghidra's launch.sh/launch.bat next to analyzeHeadless."""
        if not self.headless_path:
            return None
        support_dir = Path(self.headless_path).parent
        if platform.system() == "Windows":
            return support_dir / "launch.bat"
        return support_dir / "launch.sh"

    def _resolve_resources(self, binary_path):
        """Resolve maxmem and threads: auto-detect or use explicit values."""
        if self._maxmem_setting in (None, "auto"):
            maxmem = self.auto_maxmem(binary_path)
        else:
            maxmem = self._maxmem_setting

        if self._threads_setting in (None, "auto"):
            threads = self.auto_threads()
        else:
            threads = int(self._threads_setting)

        return maxmem, threads

    def _construct_ghidra_headless_command(self, binary_path, script_path, binary_hash,
                                           ghidra_project_dir=Path.cwd().name):

        headless = self.headless_path

        temp_script_path = Path(script_path)
        temp_script_dir = temp_script_path.parent

        if headless is None:
            binary_name = self._headless_binary_name()
            print("\n" + "="*60)
            print(f"⚠ {binary_name} not found")
            print("="*60)
            print("\nOptions:")
            print("  1. Set GHIDRA_HOME environment variable:")
            if platform.system() == "Windows":
                print("     set GHIDRA_HOME=C:\\path\\to\\ghidra")
            else:
                print("     export GHIDRA_HOME=/path/to/ghidra")
            print("  2. Use --ghidra argument:")
            print("     monocle --ghidra /path/to/ghidra ...")
            print("  3. Enter path now:")
            print("-"*60)
            user_provided_path = input("Path: ").strip('"').strip("'")

            if Path(user_provided_path).exists():
                headless = user_provided_path
                self.headless_path = headless
            else:
                raise FileNotFoundError(f"{binary_name} not found at: {user_provided_path}")

        maxmem, threads = self._resolve_resources(binary_path)
        self._resolved_maxmem = maxmem
        self._resolved_threads = threads

        app_args = [
            "-import",
            str(binary_path),
            "-scriptPath",
            str(temp_script_dir),
            "-postScript",
            str(temp_script_path.name)
        ]

        with tempfile.TemporaryDirectory() as ghidra_project_dir:
            launch_script = self._resolve_launch_script()
            use_launch = (
                launch_script and launch_script.exists()
                and (maxmem != self.DEFAULT_MAXMEM or threads != self.DEFAULT_THREADS)
            )

            if use_launch:
                gc_threads = max(2, threads // 2)
                jit_threads = max(2, threads // 4)
                vmargs = (
                    f"-XX:ParallelGCThreads={gc_threads} "
                    f"-XX:CICompilerCount={jit_threads} "
                    f"-XX:ActiveProcessorCount={threads} "
                    f"-Djava.awt.headless=true"
                )
                commandStr = [
                    str(launch_script),
                    "fg", "jdk", "Ghidra-Headless",
                    maxmem,
                    vmargs,
                    "ghidra.app.util.headless.AnalyzeHeadless",
                    str(ghidra_project_dir),
                    str(binary_hash),
                ] + app_args
            else:
                commandStr = [
                    str(headless),
                    str(ghidra_project_dir),
                    str(binary_hash),
                ] + app_args

            self._execute_blocking_command(commandStr)

    def _hash_binary(self, binary_path):
        with open(binary_path, 'rb') as f:
            binary_hash = hashlib.sha256(f.read()).hexdigest()
        return binary_hash

    def decompile_binaries_functions(self, path_to_binary, decom_folder):
        binary_hash = self._hash_binary(path_to_binary)
        with tempfile.TemporaryDirectory() as tmpdirname:
            script_path = Path(tmpdirname, "DecomScript.java").resolve()
            self.generate_ghidra_decom_script(str(decom_folder), str(script_path))
            self._construct_ghidra_headless_command(path_to_binary, script_path, binary_hash)

    def decompile_all_binaries_in_folder(self, path_to_folder, decom_folder):
        # Create a list to store all the file paths
        files_to_process = [file_path for file_path in Path(path_to_folder).iterdir() if file_path.is_file()]

        # Use a ProcessPoolExecutor to execute the decompilation in parallel
        with ProcessPoolExecutor() as executor:
            # Create a list of futures
            futures = [executor.submit(self.decompile_binaries_functions, file_path, decom_folder) for file_path in
                       files_to_process]

            # Use tqdm to show progress
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_process),
                          desc="Decompiling functions in binaries from {}".format(path_to_folder)):
                pass


if __name__ == '__main__':
    raise Exception("This is not a program entrypoint!")

