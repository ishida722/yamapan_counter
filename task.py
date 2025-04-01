import subprocess

import typer

app = typer.Typer()


@app.command()
def run_app():
    """
    Run the application.
    """
    typer.echo("Running the application...")
    # bashを実行する
    bash_command = "streamlit run src/yamapan_counter/real_time_app.py"
    subprocess.run(bash_command, shell=True)


if __name__ == "__main__":
    app()
