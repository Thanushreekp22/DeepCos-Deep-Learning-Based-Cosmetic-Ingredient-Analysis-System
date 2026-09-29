"""One-off repair: convert space-separated created_at values to ISO-8601 'T' form."""

from pymongo import MongoClient, UpdateOne


def main() -> None:
    client = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=3000)
    col = client["deepcos"]["analyses"]
    ops = []
    for doc in col.find({}, {"created_at": 1}):
        value = doc.get("created_at")
        if isinstance(value, str) and len(value) >= 19 and value[10] == " " and value[:4].isdigit():
            ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"created_at": value.replace(" ", "T", 1)}}))
    if ops:
        result = col.bulk_write(ops)
        print(f"updated {result.modified_count} document(s)")
    else:
        print("all created_at values already ISO-8601")


if __name__ == "__main__":
    main()
