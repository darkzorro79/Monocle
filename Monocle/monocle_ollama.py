import argparse
import os
import re
import tempfile
from pathlib import Path
import requests
import json

from rich.console import Console
from rich.live import Live
from rich.table import Table
from Monocle.GhidraBridge.ghidra_bridge import GhidraBridge


class MonocleOllama:
    def __init__(self, ollama_host="http://localhost:11434"):
        self.ollama_host = ollama_host
    
    def _get_code_from_decom_file(self, path_to_file):
        """Read and return the code from a decom file."""
        with open(path_to_file, "r") as file:
            return file.read()
    
    def _decompile_binary(self, decom_folder, binary):
        """Decompile the binary file and extract function information."""
        g_bridge = GhidraBridge()
        g_bridge.decompile_binaries_functions(binary, decom_folder)
        
        list_of_decom_files = []
        for file_path in Path(decom_folder).iterdir():
            binary_name, function_name, *_ = Path(file_path).name.split("__")
            list_of_decom_files.append({
                "binary_name": binary_name,
                "function_name": function_name,
                "code": self._get_code_from_decom_file(file_path)
            })
        return list_of_decom_files
    
    def _generate_dialogue_response(self, model, messages, language="English"):
        """Generate response from Ollama API."""
        url = f"{self.ollama_host}/api/chat"
        
        # Add system message for language
        system_msg = ""
        if language == "Russian":
            system_msg = "You are a code analysis expert. Always respond in Russian language (Отвечай на русском языке)."
        else:
            system_msg = "You are a code analysis expert."
        
        full_messages = [{"role": "system", "content": system_msg}] + messages
        
        payload = {
            "model": model,
            "messages": full_messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _generate_table_row(self, binary_name="", function_name="", score=0, explanation=""):
        """Generate a row for the result table."""
        return {
            "binary_name": str(binary_name),
            "function_name": function_name,
            "score": str(score),
            "explanation": str(explanation),
        }
    
    def _generate_table(self, rows, title=None):
        """Generate a table from the given rows."""
        table = Table()
        
        for column_name in rows[0].keys():
            table.add_column(str(column_name).upper().replace("_", " "))
        
        for row_dict in rows:
            table.add_row(*row_dict.values())
        
        table.caption = "Monocle + Ollama"
        
        if title:
            formatted_title = " ".join(word.capitalize() for word in title.split())
            table.title = f"[red bold underline]{formatted_title}[/red bold underline]"
        
        return table
    
    def _get_args(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Monocle with Ollama backend")
        parser.add_argument("--binary", "-b", required=True, help="The Binary to search")
        parser.add_argument("--find", "-f", required=True, help="The component to find")
        parser.add_argument("--model", "-m", 
                          default="qwen2.5-coder:7b",
                          help="Ollama model to use (default: qwen2.5-coder:7b)")
        parser.add_argument("--language", "-l",
                          default="English",
                          choices=["English", "Russian"],
                          help="Output language (default: English)")
        parser.add_argument("--ollama-host",
                          default="http://localhost:11434",
                          help="Ollama server address (default: http://localhost:11434)")
        return parser.parse_args()
    
    def entry(self):
        """Entry point of the program."""
        args = self._get_args()
        console = Console()
        
        self.ollama_host = args.ollama_host
        
        console.print(f"[cyan]Using Ollama model:[/cyan] {args.model}")
        console.print(f"[cyan]Ollama host:[/cyan] {self.ollama_host}")
        console.print(f"[cyan]Output language:[/cyan] {args.language}\n")
        
        # Test Ollama connection
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            response.raise_for_status()
            console.print("[green]✓[/green] Connected to Ollama\n")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] Cannot connect to Ollama at {self.ollama_host}")
            console.print(f"[yellow]Make sure Ollama is running: ollama serve[/yellow]")
            console.print(f"Error: {str(e)}")
            return
        
        # Check for Ghidra
        console.print("[cyan]Checking for Ghidra...[/cyan]")
        import shutil
        if shutil.which("analyzeHeadless.bat") is None:
            console.print("[yellow]⚠ Ghidra not found in PATH[/yellow]")
            console.print("Please provide the full path to analyzeHeadless.bat")
            console.print("Example: C:\\ghidra_11.2_PUBLIC\\support\\analyzeHeadless.bat")
            console.print("\nOr download Ghidra from: https://ghidra-sre.org/")
            console.print("And add it to PATH, then restart.\n")
        else:
            console.print("[green]✓[/green] Ghidra found\n")
        
        console.print("[yellow]Starting analysis...[/yellow]\n")
        
        list_of_decom_files = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            with console.status("[bold green]Decompiling binary...") as status:
                list_of_decom_files = self._decompile_binary(tmpdirname, args.binary)
                console.print("[bold green]Decompilation finished!")
                console.clear()
            
            with Live(Table(), refresh_per_second=4, console=console) as live:
                rows = []
                
                for function in list_of_decom_files:
                    binary_name = function["binary_name"]
                    function_name = function["function_name"]
                    code = function["code"]
                    
                    question = f"""You have been asked to review C decompiled code from Ghidra and identify the following '{args.find}'.

Return a score between 0 and 10, where:
- 0 means there is no indication
- 1-2 means there is something related
- 3-4 means there is a degree of evidence
- 5-6 means that there is more evidence
- 7-10 means there is significant evidence

You should be certain that the code meets these scores.

Format your response as:
- First line: single number score (0-10)
- Following lines: your explanation

Code:
{code.strip()}"""
                    
                    result = self._generate_dialogue_response(
                        args.model,
                        [{"role": "user", "content": question}],
                        args.language
                    )
                    
                    # Parse result
                    lines = result.strip().split("\n")
                    ans_number = lines[0].strip() if lines else "0"
                    explanation = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
                    
                    # Extract score number (handle formats like "3:", "Score: 3", "3", etc.)
                    score_match = re.search(r'\d+', ans_number)
                    if score_match:
                        score = int(score_match.group())
                        score = max(0, min(10, score))  # Clamp to 0-10
                    else:
                        score = 0
                        explanation = f"[Parsing error: {ans_number}] {explanation}"
                    
                    if score == 0:
                        explanation = ""
                    
                    rows.append(self._generate_table_row(
                        binary_name=binary_name,
                        function_name=function_name,
                        score=score,
                        explanation=explanation
                    ))
                    
                    # Sort by score
                    def get_score_for_sort(row):
                        try:
                            score_str = str(row['score']).replace("[green]", "").replace("[orange1]", "").replace("[red]", "")
                            return int(re.search(r'\d+', score_str).group()) if re.search(r'\d+', score_str) else 0
                        except:
                            return 0
                    
                    rows.sort(key=get_score_for_sort, reverse=True)
                    
                    # Color code scores
                    max_score = max([get_score_for_sort(row) for row in rows]) if rows else 0
                    
                    for row in rows:
                        score_val = get_score_for_sort(row)
                        if score_val >= max_score * 0.75:
                            row['score'] = f"[green]{score_val}"
                        elif score_val >= max_score * 0.5:
                            row['score'] = f"[orange1]{score_val}"
                        else:
                            row['score'] = f"[red]{score_val}"
                    
                    console.clear()
                    live.update(self._generate_table(rows, args.find))


def run():
    finder = MonocleOllama()
    finder.entry()


if __name__ == "__main__":
    run()
