import json

from src.agent_tools.base import BaseToolsAdapter, tool_method


class HotpotQATools(BaseToolsAdapter):
    @staticmethod
    def _context_chunks(row):
        context = row.get("context")
        chunks = []
        if isinstance(context, dict):
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            for idx, (title, sentences) in enumerate(zip(titles, sentences_list)):
                text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                chunks.append({"chunk_id": idx, "title": str(title), "text": text})
            return chunks
        if isinstance(context, list):
            for idx, item in enumerate(context):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    title, sentences = str(item[0]), item[1]
                    text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                    chunks.append({"chunk_id": idx, "title": title, "text": text})
                else:
                    chunks.append({"chunk_id": idx, "title": f"Chunk {idx}", "text": str(item)})
            return chunks
        return chunks

    @tool_method
    def list_passages(self, query: str):
        """List all available passages with their chunk IDs and titles.

        Args:
            query: The question to answer (passed by the agent).

        Returns:
            JSON string containing a list of objects with `chunk_id` and `title`.
        """
        chunks = self._context_chunks(self.env["row"])
        return json.dumps(
            [{"chunk_id": c["chunk_id"], "title": c["title"]} for c in chunks],
            ensure_ascii=False,
        )

    @tool_method
    def retrieve_passage_by_id(self, chunk_id: int):
        """Retrieve the full content of a passage by its chunk ID.

        Args:
            chunk_id: The zero-based ID of the passage to retrieve.

        Returns:
            JSON string containing `chunk_id`, `title`, and `content`,
            or an error object if the chunk_id is invalid.
        """
        chunks = self._context_chunks(self.env["row"])
        cid = int(chunk_id)
        if cid < 0 or cid >= len(chunks):
            return json.dumps(
                {"error": f"Invalid chunk_id={cid}", "num_chunks": len(chunks)},
                ensure_ascii=False,
            )
        c = chunks[cid]
        return json.dumps(
            {"chunk_id": c["chunk_id"], "title": c["title"], "content": c["text"]},
            ensure_ascii=False,
        )
