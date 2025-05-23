from tabulate import tabulate
import inspect
import numpy as np

def print_array(
    arr,
    headers=[],
    row_headers=[],
    title_table="",
    title_rows="",
    title_columns=""
):
    GREEN = "\033[92m"
    RESET = "\033[0m"

    # 1. Formata dados com coordenadas
    formatted_arr = []
    for i, row in enumerate(arr):
        formatted_row = [f"({i},{j}) {value}" for j, value in enumerate(row)]
        if row_headers:
            formatted_row = [row_headers[i]] + formatted_row
        formatted_arr.append(formatted_row)

    # 2. Cabeçalhos
    if headers:
        headers = [title_rows] + headers if (row_headers or title_rows) else headers
    elif row_headers:
        headers = [""] + headers

    # 3. Gera a tabela e prepara linhas
    table_str = tabulate(formatted_arr, headers=headers, tablefmt="grid")
    table_lines = table_str.splitlines()
    table_width = len(table_lines[0])

    # 4. Título da Tabela com borda superior e inferior (verde)
    if title_table:
        border_line = GREEN + "─" * table_width + RESET
        title_line = GREEN + title_table.center(table_width) + RESET
        print(border_line)
        print(title_line)
        print(border_line)

    # 5. Título das colunas com borda a partir da segunda coluna
    if title_columns:
        for line in table_lines:
            if line.startswith('+') and line.count('+') > 2:
                sep_indexes = [i for i, c in enumerate(line) if c == '+']
                break
        else:
            sep_indexes = []

        if len(sep_indexes) >= 3:
            start = sep_indexes[1] + 1
            end = sep_indexes[-1]
            col_title = f" {title_columns} "
            pad = max(0, (end - start - len(col_title)) // 2)
            padded = " " * pad + col_title + " " * (end - start - len(col_title) - pad)
            print(" " * (start - 1) + "+" + "-" * (end - start) + "+")
            print(" " * (start - 1) + "|" + padded + "|")
            print(" " * (start - 1) + "+" + "-" * (end - start) + "+")

    # 6. Imprime a tabela
    print(table_str)

def print_vars(*args):
    frame = inspect.currentframe().f_back
    values = {
        name: frame.f_locals[name]
        for name in frame.f_code.co_varnames
        if name in frame.f_locals
    }

    result = {}
    for arg in args:
        for name, val in values.items():
            if val is arg and name not in result:
                result[name] = arg
                break
        else:
            result[str(arg)] = arg

    def format_value(v, indent=2):
        pad = ' ' * indent
        if isinstance(v, dict):
            items = []
            for k, val in v.items():
                items.append(f'{pad}  "{k}": {format_value(val, indent + 2)}')
            return '{\n' + ',\n'.join(items) + f'\n{pad}}}'
        elif isinstance(v, np.ndarray):
            return f'np.array({np.array2string(v, separator=", ", prefix=" " * indent)})'
        elif isinstance(v, (list, tuple)):
            inside = ', '.join(format_value(i, indent + 2) for i in v)
            bracket = '[' if isinstance(v, list) else '('
            close = ']' if isinstance(v, list) else ')'
            return bracket + inside + close
        else:
            return repr(v)

    print("{")
    for k, v in result.items():
        type_name = type(v).__name__
        print(f'  {k} ({type_name}): {format_value(v, indent=4)},')
    print("}")

