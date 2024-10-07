class CathayDefinedException(Exception):
    error_code = "XXXX"
    error_describe = "xxxxxxxx"

    def __str__(self):
        return f"[{self.error_code}]:{self.error_describe} - {super().__str__()}"


class ParsingJsonError(CathayDefinedException):
    error_code = "1001"
    error_describe = "非 JSON 格式"


class ParsingHeaderError(CathayDefinedException):
    error_code = "1002"
    error_describe = "上行電文 Header 解析異常"


class ParsingTranRqError(CathayDefinedException):
    error_code = "1003"
    error_describe = "上行電文欄位檢核異常"


class PermissionError(CathayDefinedException):
    error_code = "2001"
    error_describe = "沒有權限訪問請求資源"


class DBInsertError(CathayDefinedException):
    error_code = "2002"
    error_describe = "Insert Database 發生異常"


class TimeOutError(CathayDefinedException):
    error_code = "2003"
    error_describe = "連線逾時"


def fmt_msg(e: Exception) -> str:
    return f"{type(e).__name__}, {e}"
