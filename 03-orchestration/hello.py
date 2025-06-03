from prefect import flow, task 

@task(retries=4,retry_delay_seconds=0.1,log_prints=True)
def Saludo():
    print("Hello World")

@flow()
def MainFlow():
    Saludo()

if __name__ == '__main__':
    MainFlow()