import argparse
import os
import re
import tempfile
from pathlib import Path

import torch
from rich.console import Console
from rich.live import Live
from rich.progress import Progress
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from Monocle.GhidraBridge.ghidra_bridge import GhidraBridge

class Monocle:
    def _load_model(self, model_name, device, hf_token=None):
        """
        Load the pre-trained language model and tokenizer.

        Args:
            model_name (str): Name of the pre-trained model.
            device (str): Device to load the model onto.
            hf_token (str, optional): HuggingFace authentication token.

        Returns:
            model (transformers.PreTrainedModel): Loaded language model.
            tokenizer (transformers.PreTrainedTokenizer): Loaded tokenizer.
        """
        
        if device == "cuda":
            # Use 4-bit quantization for GPU
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                quantization_config=quantization_config,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            # CPU mode: load with lower precision to save memory
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,
                token=hf_token,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side="left",
            token=hf_token
        )
        
        return model, tokenizer
    
    def _get_code_from_decom_file(self, path_to_file):
        """
        Read and return the code from a decom file.

        Args:
            path_to_file (str): Path to the decom file.

        Returns:
            str: Content of the decom file.
        """
        with open(path_to_file, "r") as file:
            return file.read()
        
    def _decompile_binary(self, decom_folder, binary, ghidra_path=None, maxmem=None, threads=None):
        """
        Decompile the binary file and extract function information.

        Args:
            decom_folder (str): Folder to store decompiled files.
            binary (str): Path to the binary file.
            ghidra_path (str, optional): Path to Ghidra installation.
            maxmem (str, optional): Max Java heap for Ghidra (e.g. '4G', 'auto').
            threads (str, optional): Thread count for Ghidra (e.g. '8', 'auto').

        Returns:
            list: List of dictionaries containing binary name, function name, and code.
        """
        g_bridge = GhidraBridge(ghidra_path=ghidra_path, maxmem=maxmem, threads=threads)
        g_bridge.decompile_binaries_functions(binary, decom_folder)
        
        list_of_decom_files = []

        for file_path in Path(decom_folder).iterdir():
            binary_name, function_name, *_ = Path(file_path).name.split("__")
            list_of_decom_files.append({"binary_name": binary_name, "function_name": function_name, "code": self._get_code_from_decom_file(file_path)})

        self._last_ghidra_output = getattr(g_bridge, 'last_stdout', '')
        return list_of_decom_files
        
    def _generate_dialogue_response(self, model, tokenizer, device, messages):
        """
        Generate response from the language model given the input messages.

        Args:
            model (transformers.PreTrainedModel): Loaded language model.
            tokenizer (transformers.PreTrainedTokenizer): Loaded tokenizer.
            device (str): Device to run the model on.
            messages (list): List of input messages.

        Returns:
            str: Generated response.
        """
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=200, do_sample=False, pad_token_id=50256)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return decoded[0]

    def _generate_table_row(self, binary_name="", function_name="", score=0, explanation=0):
        """
        Generate a row for the result table.

        Args:
            binary_name (str): Name of the binary file.
            function_name (str): Name of the function.
            score (int): Score assigned to the function.
            explanation (str): Explanation of the score.

        Returns:
            dict: Dictionary representing the table row.
        """
        return {
            "binary_name": str(binary_name),
            "function_name": function_name,
            "score": str(score),
            "explanation": str(explanation),
        }

    def _generate_table(self, rows, title=None):
        """
        Generate a table from the given rows.

        Args:
            rows (list): List of dictionaries representing table rows.
            title (str, optional): Title of the table. Defaults to None.

        Returns:
            rich.table.Table: Generated table.
        """
        table = Table()

        for column_name in rows[0].keys():
            table.add_column(str(column_name).upper().replace("_", " "))

        for row_dict in rows:
            table.add_row(*row_dict.values())

        table.caption = "Monocle"

        if title:
            formatted_title = " ".join(word.capitalize() for word in title.split())
            table.title = f"[red bold underline]{formatted_title}[/red bold underline]"

        return table

    def _get_args(self):
        """
        Parse command line arguments.

        Returns:
            argparse.Namespace: Parsed arguments.
        """
        parser = argparse.ArgumentParser(description="Local Language Model (LLM) - Explain code snippets")
        parser.add_argument("--binary", "-b", required=True, help="The Binary to search")
        parser.add_argument("--find", "-f", required=True, help="The component to find")
        parser.add_argument("--token", "-t", help="HuggingFace authentication token (or set HF_TOKEN env variable)")
        parser.add_argument("--model", "-m", 
                          default="mistralai/Mistral-7B-Instruct-v0.2",
                          help="HuggingFace model to use (default: mistralai/Mistral-7B-Instruct-v0.2)")
        parser.add_argument("--language", "-l",
                          default="English",
                          choices=["English", "Russian"],
                          help="Output language (default: English)")
        parser.add_argument("--ghidra", "-g",
                          help="Path to Ghidra installation (or set GHIDRA_HOME env variable)")
        parser.add_argument("--maxmem",
                          default="auto",
                          help="Max Java heap for Ghidra (default: auto). Examples: 4G, 8G, auto")
        parser.add_argument("--threads",
                          default="auto",
                          help="CPU threads for Ghidra analysis (default: auto). Examples: 4, 8, auto")
        return parser.parse_args()
    
    def _remove_inst_tags(self, text):
        """
        Remove instruction tags from the given text.

        Args:
            text (str): Input text containing instruction tags.

        Returns:
            str: Text with instruction tags removed.
        """
        pattern = r'\[INST\].*?\[/INST\]'
        clean_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return clean_text.replace("<s>", "").replace("</s>", "").replace("Explanation:", "").strip()

    def entry(self):
        """
        Entry point of the program.
        """
        args = self._get_args()
        console = Console()
        
        device = "cpu"
        if torch.cuda.is_available():
            try:
                test_tensor = torch.rand(2, 2).cuda()
                _ = test_tensor * 2
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                console.print(f"[green]✓[/green] Using GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            except RuntimeError:
                cap = torch.cuda.get_device_capability(0)
                console.print(f"[yellow]⚠ GPU detected but incompatible: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})[/yellow]")
                console.print("[yellow]  Ensure PyTorch is installed with matching CUDA version:[/yellow]")
                console.print("[yellow]  pip install torch --index-url https://download.pytorch.org/whl/cu128[/yellow]")
                console.print("[yellow]  Falling back to CPU mode\n[/yellow]")
        else:
            console.print("[yellow]⚠ No CUDA device found, using CPU (will be slower)[/yellow]\n")
        
        model_name = args.model
        output_language = args.language
        console.print(f"[cyan]Using model:[/cyan] {model_name}")
        console.print(f"[cyan]Output language:[/cyan] {output_language}\n")
        
        # Get HuggingFace token from args or environment variable
        hf_token = args.token or os.environ.get("HF_TOKEN")
        
        if not hf_token:
            console.print("[bold red]Error:[/bold red] HuggingFace token required!")
            console.print("Please either:")
            console.print("  1. Set HF_TOKEN environment variable")
            console.print("  2. Use --token parameter")
            console.print("\nGet your token at: https://huggingface.co/settings/tokens")
            console.print(f"Request access to model at: https://huggingface.co/{model_name}")
            return
        
        try:
            ghidra_test = GhidraBridge(ghidra_path=args.ghidra)
            if ghidra_test.headless_path:
                console.print(f"[green]✓[/green] Ghidra found: {ghidra_test.headless_path}\n")
            else:
                console.print("[yellow]⚠ Ghidra not found in PATH or GHIDRA_HOME[/yellow]")
                console.print("[yellow]  You will be prompted for the path during analysis.[/yellow]")
                console.print("[yellow]  Tip: use --ghidra or set GHIDRA_HOME to avoid this.\n[/yellow]")
        except FileNotFoundError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return
        
        if device == "cpu":
            console.print("[yellow]Loading model in CPU mode, this may take several minutes and requires 32GB+ RAM...[/yellow]")
        
        model, tokenizer = self._load_model(model_name, device, hf_token)
        console.clear()
        
        list_of_decom_files = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            with console.status("[bold green]Decompiling binary..."):
                try:
                    list_of_decom_files = self._decompile_binary(tmpdirname, args.binary, ghidra_path=args.ghidra, maxmem=args.maxmem, threads=args.threads)
                except RuntimeError as e:
                    console.print(f"[bold red]Ghidra error:[/bold red] {e}")
                    return

            if not list_of_decom_files:
                console.print("[bold red]Error:[/bold red] No functions were decompiled from the binary.")
                console.print("[yellow]Possible causes:[/yellow]")
                console.print("  - Binary format is not supported by Ghidra")
                console.print("  - Ghidra failed to analyze the binary")
                console.print("  - Binary is packed or obfuscated\n")
                ghidra_log = getattr(self, '_last_ghidra_output', '')
                if ghidra_log:
                    last_lines = "\n".join(ghidra_log.splitlines()[-30:])
                    console.print("[dim]--- Ghidra output (last 30 lines) ---[/dim]")
                    console.print(f"[dim]{last_lines}[/dim]")
                return

            console.print(f"[green]✓[/green] Decompiled [bold]{len(list_of_decom_files)}[/bold] functions\n")

            with Live(Table(), refresh_per_second=4, console=console) as live:
                rows = []    

                for function in list_of_decom_files:
                    binary_name = function["binary_name"]
                    function_name = function["function_name"]
                    code = function["code"]

                    language_instruction = ""
                    if output_language == "Russian":
                        language_instruction = " Answer in Russian language (Отвечай на русском языке)."
                    
                    question = f"You have been asked to review C decompiled code from Ghidra and identify the following '{args.find}'. Return a score between 0 and 10, where 0 means there is no indication, 1 to 2 means there is something related, 3 to 4 means there is a degree of evidence, 5 to 6 means that there is more evidence, and 7 to 10 means there is significant evidence. You should be certain that the code meets these scores. Format your response as a single number score, followed my a new line, followed by your explanation.{language_instruction} \n Code: \n {code.strip()}"
                    
                    result = self._generate_dialogue_response(model, tokenizer, device, [{"role": "user", "content": question}])
                    result = self._remove_inst_tags(result)

                    ans_number, *explanation = result.split("\n")
                    explanation = "".join(explanation).strip()

                    # Extract score number (handle formats like "3:", "Score: 3", "3", etc.)
                    score_match = re.search(r'\d+', ans_number)
                    if score_match:
                        score = int(score_match.group())
                        # Clamp score to 0-10 range
                        score = max(0, min(10, score))
                    else:
                        # If no number found, default to 0
                        score = 0
                        explanation = f"[Parsing error: {ans_number}] {explanation}"

                    if score == 0:
                        explanation = ""

                    rows.append(self._generate_table_row(binary_name=binary_name, function_name=function_name, score=score, explanation=explanation))

                    for row_dict in rows:
                        words_to_replace = ["[green]", "[orange1]", "[red]"]
                        score_value = str(row_dict["score"])
                        for word in words_to_replace:
                            score_value = score_value.replace(word, "")
                        row_dict["score"] = score_value

                    # Sort by score (handle both string and int scores)
                    def get_score_for_sort(row):
                        try:
                            score_str = str(row['score']).replace("[green]", "").replace("[orange1]", "").replace("[red]", "")
                            return int(re.search(r'\d+', score_str).group()) if re.search(r'\d+', score_str) else 0
                        except:
                            return 0
                    
                    rows.sort(key=get_score_for_sort, reverse=True)
                    
                    # Get max score safely
                    max_score = max([get_score_for_sort(row) for row in rows]) if rows else 0
                    
                    for row in rows:
                        score = get_score_for_sort(row)
                        if score >= max_score * 0.75:
                            row['score'] = f"[green]{score}"
                        elif score >= max_score * 0.5:
                            row['score'] = f"[orange1]{score}"
                        else:
                            row['score'] = f"[red]{score}"
                    
                    console.clear()
                    live.update(self._generate_table(rows, args.find))

def run():
    finder = Monocle()
    finder.entry()

if __name__ == "__main__":
    run()
