# backend/parsers/code_cleaner.py
import ast

class ExecutableStripper(ast.NodeTransformer):
    """
    상위 레벨에서 실행되는 코드들을 제거한다.
    - if __name__ == '__main__': 블록
    - top-level function call
    - top-level assignment + call
    - for/while loop, with block 등 실행문
    """

    def visit_If(self, node):
        # if __name__ == "__main__": 제거
        try:
            if (isinstance(node.test, ast.Compare) and
                isinstance(node.test.left, ast.Name) and
                node.test.left.id == "__name__" and
                any(isinstance(op, ast.Eq) for op in node.test.ops) and
                any(isinstance(v, ast.Constant) and v.value == "__main__"
                    for v in node.test.comparators)):
                return None
        except:
            pass

        return self.generic_visit(node)

    def visit_Expr(self, node):
        # print(), train(), evaluate(), model(...), 등 실행문 제거
        if isinstance(node.value, ast.Call):
            return None
        return node

    def visit_For(self, node):
        # for 문 실행 제거
        return None

    def visit_While(self, node):
        # while 문 실행 제거
        return None

    def visit_With(self, node):
        # with open() as f: 같은 실행 제거
        return None

    def visit_Assign(self, node):
        # data = load_data() 같은 실행 제거
        if isinstance(node.value, ast.Call):
            return None
        return node


def strip_executable_code(code_text: str) -> str:
    """
    주어진 코드 문자열에서 실행되는 구문만 제거한 안전한 코드로 변환한다.
    """
    tree = ast.parse(code_text)
    stripper = ExecutableStripper()
    cleaned = stripper.visit(tree)
    ast.fix_missing_locations(cleaned)
    return ast.unparse(cleaned)
