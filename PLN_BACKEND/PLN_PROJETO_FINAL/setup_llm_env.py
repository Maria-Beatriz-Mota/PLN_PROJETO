import argparse
import importlib
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from dotenv import load_dotenv

MODEL_NAME_LLM_HF = "meta-llama/Llama-3.2-3B-Instruct"
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_KEY")
REQUIRED_PACKAGES = {
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "huggingface_hub": "0.33.0",
    "accelerate": "1.8.1",
    "sentencepiece": "0.2.0",
}
OPTIONAL_PACKAGES = {
    "torch": None,
    "safetensors": None,
}


def load_local_env() -> Path | None:
    env_path = Path(__file__).resolve().with_name(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)
        return env_path
    return None


def get_installed_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None



def ensure_package_import(package_name: str) -> bool:
    module_name = package_name.replace("-", "_")
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False



def build_install_args() -> list[str]:
    packages = []
    for package_name, version in REQUIRED_PACKAGES.items():
        installed_version = get_installed_version(package_name)
        if installed_version != version:
            packages.append(f"{package_name}=={version}" if version else package_name)
    for package_name, version in OPTIONAL_PACKAGES.items():
        if get_installed_version(package_name) is None:
            packages.append(f"{package_name}=={version}" if version else package_name)
    return packages



def install_packages() -> int:
    packages = build_install_args()
    if not packages:
        print("Dependencias principais ja estao alinhadas com as versoes esperadas.")
        return 0
    command = [sys.executable, "-m", "pip", "install", *packages]
    print("Instalando dependencias da LLM...")
    print(" ".join(command))
    process = subprocess.run(command, text=True)
    return process.returncode



def read_hf_token() -> tuple[str | None, str | None]:
    for env_name in TOKEN_ENV_VARS:
        token = os.environ.get(env_name)
        if token:
            if env_name != "HF_TOKEN" and not os.environ.get("HF_TOKEN"):
                os.environ["HF_TOKEN"] = token
            return env_name, token
    return None, None



def print_dependency_report() -> None:
    print("=== Dependencias da LLM ===")
    for package_name, expected_version in REQUIRED_PACKAGES.items():
        installed_version = get_installed_version(package_name)
        if not installed_version:
            status = "AUSENTE"
        elif expected_version and installed_version != expected_version:
            status = "VERSAO DIFERENTE"
        else:
            status = "OK"
        expected_label = expected_version or "qualquer"
        print(
            f"- {package_name}: instalado={installed_version or 'nao'} | esperado={expected_label} | status={status}"
        )

    print("=== Dependencias opcionais ===")
    for package_name in OPTIONAL_PACKAGES:
        installed_version = get_installed_version(package_name)
        status = "OK" if installed_version else "AUSENTE"
        print(f"- {package_name}: instalado={installed_version or 'nao'} | status={status}")



def print_token_guidance() -> None:
    env_name, token = read_hf_token()
    print("=== Token Hugging Face ===")
    if token:
        suffix = token[-4:] if len(token) >= 4 else token
        print(f"- Token encontrado em {env_name} (final ...{suffix})")
        print("- O notebook pode reutilizar esse token na sessao atual.")
        return

    print("- Nenhum token encontrado no ambiente atual.")
    print("- Se voce usar arquivo .env, deixe-o na mesma pasta deste script/notebook.")
    print("- Variaveis aceitas no .env: HF_TOKEN, HUGGINGFACEHUB_API_TOKEN ou HUGGINGFACE_API_KEY")
    print("- Defina o token no PowerShell antes de abrir ou reiniciar o kernel:")
    print('  $env:HF_TOKEN="SEU_TOKEN_AQUI"')
    print("- Alternativa equivalente:")
    print('  $env:HUGGINGFACEHUB_API_TOKEN="SEU_TOKEN_AQUI"')
    print("- Exemplo em .env:")
    print('  HF_TOKEN="SEU_TOKEN_AQUI"')



def test_model_access(model_name: str) -> int:
    env_name, token = read_hf_token()
    if not token:
        print("Nao foi possivel testar acesso ao modelo: token ausente.")
        return 1

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        info = api.model_info(model_name)
        print(f"Acesso ao modelo confirmado: {info.id}")
        print(f"Token lido de: {env_name}")
        return 0
    except Exception as exc:
        print(f"Falha ao consultar o modelo '{model_name}': {exc}")
        print("Verifique se a licenca do modelo foi aceita e se o token tem permissao de leitura.")
        return 1



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Checa dependencias e token usados pelo notebook da LLM via Hugging Face."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Instala as dependencias principais da LLM no interpretador atual.",
    )
    parser.add_argument(
        "--check-model-access",
        action="store_true",
        help="Testa se o token atual consegue acessar o modelo configurado.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME_LLM_HF,
        help="Modelo da Hugging Face usado no teste de acesso.",
    )
    args = parser.parse_args()

    env_path = load_local_env()
    if env_path:
        print(f"Arquivo .env carregado: {env_path}")
    else:
        print("Arquivo .env nao encontrado ao lado do script. Seguindo apenas com variaveis do ambiente atual.")

    print_dependency_report()
    print()
    print_token_guidance()

    if args.install:
        exit_code = install_packages()
        if exit_code != 0:
            return exit_code
        print("Instalacao concluida. Reinicie o kernel antes de rerodar o notebook.")

    if args.check_model_access:
        print()
        return test_model_access(args.model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
