from tabulate import tabulate

def print_array(arr, headers=[], row_headers=[], title=None):
    GREEN = "\033[92m"
    RESET = "\033[0m"

    formatted_arr = []
    for i, row in enumerate(arr):
        formatted_row = [f"[{i},{j}] {value}" for j, value in enumerate(row)]
        if row_headers:
            formatted_row = [row_headers[i]] + formatted_row
        formatted_arr.append(formatted_row)

    if row_headers:
        headers = [""] + (headers or [])

    table_str = tabulate(formatted_arr, headers=headers, tablefmt="grid")

    if title:
        # Conta o número de colunas da tabela (baseado na primeira linha com +----+)
        num_columns = table_str.splitlines()[0].count("+") - 1
        table_width = len(table_str.splitlines()[0])
        title_line = GREEN + title.center(table_width) + RESET
        print(title_line)

    print(table_str)


