import pytest
from unittest.mock import MagicMock
import time

# 假設 app.py 在上一層目錄，這裡簡單引入目標函式
# 實務上建議將 app.py 內的核心邏輯拆分到 utils.py 以利測試
from app import execute_with_retry

def test_execute_with_retry_success_first_try():
    mock_func = MagicMock(return_value="success")
    result = execute_with_retry(mock_func)
    assert result == "success"
    assert mock_func.call_count == 1

def test_execute_with_retry_retryable_error():
    # 模擬前兩次發生 429 錯誤，第三次成功
    mock_func = MagicMock(side_effect=[Exception("HTTP 429 Too Many Requests"), Exception("503 Service Unavailable"), "finally_success"])
    
    # 為了加速測試，我們可以暫時 patch 掉 time.sleep
    result = execute_with_retry(mock_func, max_retries=5, base_delay=0.01)
    
    assert result == "finally_success"
    assert mock_func.call_count == 3

def test_execute_with_retry_non_retryable_error():
    # 模擬發生不可重試的錯誤 (例如 ValueError)
    mock_func = MagicMock(side_effect=ValueError("Invalid Input"))
    
    with pytest.raises(ValueError):
        execute_with_retry(mock_func, max_retries=3)
    
    # 發生非預期的錯誤應該直接拋出，不該重試
    assert mock_func.call_count == 1
