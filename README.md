# UniHelp — University FAQ Assistant (RAG)

เว็บแชตบอตตัวอย่างสำหรับตอบคำถามที่พบบ่อยของนักศึกษา โดยค้นคืนจากเอกสาร TXT ด้วย multilingual Sentence Transformers และ FAISS ก่อนส่ง Context ให้ Groq LLM สรุปคำตอบ

## Domain
ลงทะเบียนเรียน, เพิ่มถอนรายวิชา, ปฏิทินการศึกษา, ค่าธรรมเนียม, เอกสารรับรอง, อาจารย์ที่ปรึกษา, LMS, ห้องสมุด, ทุนการศึกษา และการสำเร็จการศึกษา

## หมายเหตุสำคัญ
เอกสารใน `data/` เป็นข้อมูลตัวอย่างที่เรียบเรียงเพื่อการศึกษา ไม่ใช่ระเบียบหรือประกาศของมหาวิทยาลัยจริง วันเวลา จำนวนเงิน แบบฟอร์ม และช่องทางติดต่อจริงต้องตรวจจากสถาบันของผู้ใช้ ระบบถูกออกแบบให้ไม่เดาข้อมูลเฉพาะที่ไม่มีใน Context

## วิธีทำงาน
1. อ่านไฟล์ `.txt` ใน `data/` และทำความสะอาดช่องว่าง
2. แบ่งเป็น Chunk ที่มี overlap
3. ใช้ `paraphrase-multilingual-MiniLM-L12-v2` สร้าง Embedding
4. เก็บเวกเตอร์ใน FAISS และค้น Top-k ด้วย cosine similarity (เวกเตอร์ normalize + inner product)
5. ส่ง Context พร้อมคำถามให้ Groq LLM ผ่าน Prompt ที่จำกัดแหล่งข้อมูล
6. แสดงคำตอบและไฟล์อ้างอิง

## Run locally
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
กำหนดตัวแปร `GROQ_API_KEY` ใน Environment หรือเพิ่มใน `.streamlit/secrets.toml` (ห้าม commit ไฟล์นี้)
```toml
GROQ_API_KEY = "YOUR_API_KEY"
```
จากนั้น:
```bash
streamlit run app.py
```

## GitHub + Streamlit Community Cloud
1. สร้าง GitHub repository ใหม่และอัปโหลดไฟล์ทั้งหมด ยกเว้น secrets จริง
2. เข้า https://share.streamlit.io/ แล้วเลือก Create app
3. เลือก Repository, Branch และ Main file `app.py`
4. เปิด App settings > Secrets แล้วใส่:
```toml
GROQ_API_KEY = "YOUR_API_KEY"
```
5. Deploy รอ build และเปิด URL ทดสอบใน Incognito

## Prompt ตัวอย่าง
“คุณคือ UniHelp ผู้ช่วยตอบคำถามนักศึกษา ใช้เฉพาะ CONTEXT ที่ให้มา ห้ามเดาวันที่ จำนวนเงิน หรือระเบียบ หากข้อมูลไม่พอให้แจ้งว่าไม่พบข้อมูลที่เพียงพอ และระบุชื่อไฟล์อ้างอิง”

## ชุดทดสอบ
`test_questions.csv` มี 12 คำถามพร้อม expected answer/status รวมคำถามนอกขอบเขต 2 ข้อ
