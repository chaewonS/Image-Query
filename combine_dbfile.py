# python integrate_h5_to_db.py 실행 후, 두 개의 db를 합치는 코드
import sqlite3
import h5py

# 기존 DB 파일 경로
db_file_path = '/home/ubuntu/cw/Hierarchical-Localization/outputs/integrate/combined_refer.db'

# 300ms_reference_random.db 파일 경로
reference_db_file_path = '/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/300ms_reference_random.db'

# combined_final.db 파일 경로
combined_final_db_file_path = '/home/ubuntu/cw/Hierarchical-Localization/outputs/integrate/refer_combined_final.db'

# .db 파일을 열어 연결
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# final 데이터베이스에 연결하고 테이블 생성
final_conn = sqlite3.connect(combined_final_db_file_path)
final_cursor = final_conn.cursor()
final_cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        filename TEXT PRIMARY KEY,
        global_descriptor BLOB,
        histogram BLOB
    )
''')

# 300ms_reference_random.db에서 데이터 읽기
reference_conn = sqlite3.connect(reference_db_file_path)
reference_cursor = reference_conn.cursor()
reference_cursor.execute("SELECT filename, histogram FROM images")

# 기존 DB 데이터를 읽고 새로운 DB에 추가
for row in reference_cursor:
    filename = row[0]
    histogram = row[1]
    cursor.execute("SELECT global_descriptor FROM images WHERE filename=?", (filename,))
    global_descriptor = cursor.fetchone()
    if global_descriptor is not None:
        final_cursor.execute("INSERT INTO images (filename, global_descriptor, histogram) VALUES (?, ?, ?)", (filename, global_descriptor[0], histogram))

# 변경 사항 저장 및 연결 닫기
final_conn.commit()
final_conn.close()
reference_conn.close()
conn.close()

print("새로운 combined_final.db 파일이 성공적으로 생성되었습니다.")
