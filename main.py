from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

from services.fpgrowth_services import run_fpgrowth_analysis


app = FastAPI(
    title="API Analisis FP-Growth",
    description="API untuk upload dataset mentah dan menjalankan analisis association rules menggunakan FP-Growth",
    version="1.0.0"
)

# Supaya API bisa dipanggil dari Laravel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output/api_result"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {
        "status": "success",
        "message": "API FP-Growth aktif",
        "docs": "http://127.0.0.1:8000/docs"
    }


@app.post("/api/fpgrowth/analyze")
async def analyze_fpgrowth(
    file: UploadFile = File(...),
    min_support: float = Form(0.01),
    min_confidence: float = Form(0.5),
    min_lift: float = Form(0.0),
    include_operator: bool = Form(True),
    include_waktu: bool = Form(True),
    only_product_rules: bool = Form(False),
    top_n: int = Form(20),
):
    try:
        # ==========================
        # VALIDASI FILE
        # ==========================
        original_filename = file.filename

        if original_filename is None or original_filename == "":
            return {
                "status": "error",
                "message": "File belum dipilih."
            }

        if not original_filename.lower().endswith((".xls", ".xlsx")):
            return {
                "status": "error",
                "message": f"File {original_filename} bukan file Excel. Gunakan file .xls atau .xlsx."
            }

        # ==========================
        # SIMPAN FILE UPLOAD
        # ==========================
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        contents = await file.read()

        with open(file_path, "wb") as f:
            f.write(contents)

        # ==========================
        # JALANKAN FP-GROWTH
        # ==========================
        result = run_fpgrowth_analysis(
            file_paths=file_path,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            include_operator=include_operator,
            include_waktu=include_waktu,
            only_product_rules=only_product_rules,
            top_n=top_n,
            output_dir=OUTPUT_DIR,
            save_output=True,
            save_intermediate=False
        )

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }