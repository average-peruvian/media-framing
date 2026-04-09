import json
from ..distrib import BaseWorker

class EntityGrifter(BaseWorker):
    def process_row(self, row) -> dict | None:
        for ent in json.loads(row["entities"]):
            self._buffer.append({
                "id":         row["id"],
                "local_id":   ent["id"],
                "mention_id": f"{row['id']}__{ent['id']}",
                "entity_id":  None,
            })

        return None

    def on_error(self, row, exc: Exception) -> dict | None:
        print(f"✗ {row['id']}: {exc}")
        return None

class RelationGrifter(BaseWorker):
    def process_row(self, row) -> dict | None:
        for rel in json.loads(row["relations"]):
            self._buffer.append({
                "relation_id": f"{row['id']}__{rel['id']}",
                "id":          row["id"],
                "local_id":    rel["id"],
                "subject":     rel["subject"],
                "object":      rel["object"],
                "relation":    rel["relation"],
                "evidence":    rel["evidence"],
                "confidence":  rel["confidence"],
                "type_id":     None,
            })

        return None

    def on_error(self, row, exc: Exception) -> dict | None:
        print(f"✗ {row['id']}: {exc}")
        return None

class ExplodeEventsWorker(BaseWorker):

    def process_row(self, row) -> dict | None:
        for evt in json.loads(row["events"]):
            self._buffer.append({
                "event_id":   f"{row['id']}__{evt['id']}",
                "id":         row["id"],
                "local_id":   evt["id"],
                "event_type": evt["type"],
                "trigger":    evt["trigger"],
                "confidence": evt["confidence"],
                "type_id":    None,
            })

        return None

    def on_error(self, row, exc: Exception) -> dict | None:
        print(f"✗ {row['id']}: {exc}")
        return None
