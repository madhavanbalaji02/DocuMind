"""Convert Textract TABLE blocks into clean Markdown tables."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Cell:
    row: int
    col: int
    text: str
    is_header: bool = False
    row_span: int = 1
    col_span: int = 1


class TableToMarkdown:
    """Reconstruct Textract TABLE blocks as clean Markdown."""

    def convert(self, table_block: dict, block_map: dict[str, dict]) -> str:
        cells = self._extract_cells(table_block, block_map)
        if not cells:
            return ""

        max_row = max(c.row for c in cells)
        max_col = max(c.col for c in cells)

        # Build grid
        grid: list[list[str]] = [[""] * (max_col + 1) for _ in range(max_row + 1)]
        header_rows: set[int] = set()
        for cell in cells:
            grid[cell.row][cell.col] = cell.text
            if cell.is_header:
                header_rows.add(cell.row)

        lines: list[str] = []
        for row_idx, row in enumerate(grid):
            row_str = "| " + " | ".join(cell or "" for cell in row) + " |"
            lines.append(row_str)
            if row_idx == 0 or row_idx in header_rows:
                sep = "| " + " | ".join("-" * max(3, len(cell or "")) for cell in row) + " |"
                lines.append(sep)

        return "\n".join(lines)

    def _extract_cells(self, table_block: dict, block_map: dict[str, dict]) -> list[Cell]:
        cells: list[Cell] = []
        for rel in table_block.get("Relationships", []):
            if rel["Type"] != "CHILD":
                continue
            for child_id in rel["Ids"]:
                block = block_map.get(child_id, {})
                if block.get("BlockType") != "CELL":
                    continue
                row = block.get("RowIndex", 1) - 1  # 0-indexed
                col = block.get("ColumnIndex", 1) - 1
                is_header = block.get("EntityTypes", []) and "COLUMN_HEADER" in block.get("EntityTypes", [])
                text = self._cell_text(block, block_map)
                cells.append(Cell(
                    row=row,
                    col=col,
                    text=text,
                    is_header=is_header,
                    row_span=block.get("RowSpan", 1),
                    col_span=block.get("ColumnSpan", 1),
                ))
        return cells

    def _cell_text(self, cell_block: dict, block_map: dict[str, dict]) -> str:
        words: list[str] = []
        for rel in cell_block.get("Relationships", []):
            if rel["Type"] != "CHILD":
                continue
            for child_id in rel["Ids"]:
                child = block_map.get(child_id, {})
                if child.get("BlockType") == "WORD":
                    words.append(child.get("Text", ""))
        return " ".join(words)
