from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from app.domain.enums import MessageRole
from app.infrastructure.db.store import SqlAlchemyConversationStore


@pytest.mark.asyncio
async def test_messages_are_persisted_in_sequence(tmp_path: Path) -> None:
    database_path = (tmp_path / "paperpilot-test.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    store = SqlAlchemyConversationStore(engine)
    await store.initialize()

    conversation = await store.create_conversation("Test")
    first = await store.append_message(conversation.id, MessageRole.USER, "Find RAG papers")
    second = await store.append_message(conversation.id, MessageRole.ASSISTANT, "Searching")
    messages = await store.list_messages(conversation.id)

    assert first.sequence == 1
    assert second.sequence == 2
    assert [message.content for message in messages] == ["Find RAG papers", "Searching"]
    await store.close()

