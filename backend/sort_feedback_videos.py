"""
sort_feedback_videos.py
Sorts saved_videos/ into subfolders based on feedback:
  saved_videos/correct/FAKE/ or correct/REAL/
  saved_videos/wrong/FAKE/   or wrong/REAL/
Run: python sort_feedback_videos.py
"""

import os, sqlite3, shutil

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "feedback.db")
VIDEOS_DIR  = os.path.join(BASE_DIR, "saved_videos")

# Create destination folders
for fb in ["correct", "wrong"]:
    for label in ["FAKE", "REAL"]:
        os.makedirs(os.path.join(VIDEOS_DIR, fb, label), exist_ok=True)

con  = sqlite3.connect(DB_PATH)
rows = con.execute(
    "SELECT video_path, true_label, feedback FROM feedback WHERE video_path IS NOT NULL"
).fetchall()
con.close()

moved   = 0
skipped = 0

for path, true_label, feedback in rows:
    if not os.path.exists(path):
        print(f"  SKIP (not found): {os.path.basename(path)}")
        skipped += 1
        continue

    dest_dir  = os.path.join(VIDEOS_DIR, feedback, true_label)
    dest_path = os.path.join(dest_dir, os.path.basename(path))

    if os.path.abspath(path) == os.path.abspath(dest_path):
        print(f"  ALREADY SORTED: {os.path.basename(path)}")
        continue

    shutil.move(path, dest_path)
    print(f"  [{feedback.upper()} / {true_label}] {os.path.basename(path)}")
    moved += 1

print(f"\nDone. Moved: {moved}  |  Skipped: {skipped}")
print(f"\nStructure:")
print(f"  saved_videos/correct/FAKE/  ← model correctly identified fakes")
print(f"  saved_videos/correct/REAL/  ← model correctly identified reals")
print(f"  saved_videos/wrong/FAKE/    ← these are actually FAKE (model said REAL)")
print(f"  saved_videos/wrong/REAL/    ← these are actually REAL (model said FAKE)")
