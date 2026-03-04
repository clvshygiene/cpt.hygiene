import pytest
from unittest.mock import patch, MagicMock

# 同樣先引入目標函式
from app import claim_notion_task

@patch('app.get_notion_client')
def test_claim_notion_task_no_client(mock_get_client):
    # 模擬 client 為 None 的邊界情況
    mock_get_client.return_value = None
    success, msg = claim_notion_task("page_123", "112001")
    
    assert success is False
    assert "Notion 服務目前未啟用" in msg

@patch('app.get_notion_client')
def test_claim_notion_task_duplicate(mock_get_client):
    # 模擬已有學生認領的情況
    mock_client = MagicMock()
    mock_client.pages.retrieve.return_value = {
        "properties": {
            "需求人數": {"number": 2},
            "認領學號": {"rich_text": [{"text": {"content": "112001, 112002"}}]}
        }
    }
    mock_get_client.return_value = mock_client
    
    success, msg = claim_notion_task("page_123", "112001")
    assert success is False
    assert "已經認領過此任務囉" in msg
