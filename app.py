from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
import uuid
from run import run_squat_analysis
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import time
import json
from pathlib import Path
from run_realtime import gen_video_stream, get_latest_analysis_info
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy import select, delete, Column, Integer, Float, String, DateTime, func, Index
from flask_migrate import Migrate
import requests
from pathlib import Path
import os, joblib, torch, numpy as np
from model.ft_transformer import FTTransformer 

app = Flask(__name__, static_folder="static")
app.jinja_env.filters['from_json'] = json.loads
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///squat_analysis1.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db, compare_type=True)
UPLOAD_FOLDER = 'uploaded_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Base = db.Model

api_url = "http://localhost/v1/chat-messages"  # Dify 本地部署 API 端點
api_key = "app-lAcz4GKJGnPp1w7rrbmUl8jg"  # 你的金鑰

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
# 資料庫中「分析紀錄」的結構
class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), nullable=False)
    squat_type = db.Column(db.String(32), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    report_json = db.Column(db.Text, nullable=False)
    def __repr__(self):
        return f'<AnalysisHistory {self.id}>'

class UserSummary(Base):
    __tablename__ = "user_summaries"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(64), index=True, nullable=False)
    avg_score = db.Column(db.Float, nullable=False)
    squat_count = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

Index("idx_username_created_desc", UserSummary.username, UserSummary.created_at.desc())


APP_DIR = Path(__file__).resolve().parent
W_PATH  = Path(os.environ.get("W_PATH", APP_DIR / "model_weights.pt"))
M_PATH  = Path(os.environ.get("M_PATH", APP_DIR / "preprocess_meta.pkl"))


_device_env = os.environ.get("DEVICE", "auto").lower()
if _device_env == "cuda" and torch.cuda.is_available():
    DEVICE = "cuda"
elif _device_env in ("cpu", "cuda"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] CWD={os.getcwd()}")
print(f"[INFO] APP_DIR={APP_DIR}")
print(f"[INFO] W_PATH={W_PATH}")
print(f"[INFO] M_PATH={M_PATH}")
print(f"[INFO] DEVICE={DEVICE} (cuda_available={torch.cuda.is_available()})")

model = None
ckpt  = None
CATEG_COLS = []
CONT_COLS  = []
CAT_MAPS   = {}
SCALER_MEAN = None
SCALER_SCALE = None
NUM_CLASSES = None

def _load_model_once():
    global model, ckpt_meta
    if model is not None:
        return
    if not (W_PATH.exists() and M_PATH.exists()):
        print("[WARN] 找不到 model_weights.pt 或 preprocess_meta.pkl，/predict 將回 503")
        return

    print(f"[INFO] loading weights (safe): {W_PATH}")
    state_dict = torch.load(str(W_PATH), map_location=DEVICE, weights_only=True)

    ckpt_meta = joblib.load(str(M_PATH))
    assert ckpt_meta.get("model_type") == "ft", "此服務僅支援 FTTransformer 的 checkpoint"

    model_ = FTTransformer(
        categories=tuple(ckpt_meta["categories"]),
        num_continuous=int(ckpt_meta["num_continuous"]),
        dim=32, depth=3, heads=4, dim_out=int(ckpt_meta["num_classes"])
    ).to(DEVICE)
    model_.load_state_dict(state_dict, strict=True)
    model_.eval()
    model = model_

def _encode_categorical(payload: dict) -> torch.Tensor:
    vals = []
    for col in ckpt_meta["categ_cols"]:
        mapping = ckpt_meta["cat_maps"][col]
        vals.append(mapping.get(payload.get(col), 0))  # 未知值→0
    return torch.tensor(vals, dtype=torch.long).unsqueeze(0)

def _to_float_or_none(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        return float(x)
    except Exception:
        return None
    
def _scale_continuous(payload: dict) -> torch.Tensor:
    arr = np.array([float(payload[col]) for col in ckpt_meta["cont_cols"]], dtype=np.float32)
    mu  = ckpt_meta.get("scaler_mean")
    sc  = ckpt_meta.get("scaler_scale")
    if (mu is not None) and (sc is not None):
        arr = (arr - np.array(mu, dtype=np.float32)) / np.array(sc, dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

def _softmax(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

# --------------------------
# 主頁面：上傳影片
# --------------------------
@app.route('/')
def index():
    # 確保模型在首頁就載入
    _load_model_once()
    return render_template('upload.html')

# --------------------------
# 分析影片動作
# --------------------------
@app.route('/analyze_ajax', methods=['POST'])
def analyze_video_ajax():
    video_file = request.files['video']
    username = request.form['username']
    squat_type = request.form["squat_type"]

    # 儲存使用者上傳的影片
    filename = str(uuid.uuid4()) + ".mp4"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    # 執行分析，現在會回傳 conversation_id
    llm_answer, report_table, result_video_path, error_image_paths, average_score, squat_count, conversation_id = run_squat_analysis(video_path, squat_type)

    # 儲存分析結果到資料庫
    history_entry = AnalysisHistory(
        username=username,
        squat_type=squat_type,
        report_json=json.dumps({
            "report_table": report_table,
            "average_score": average_score,
            "squat_count": squat_count
        }, ensure_ascii=False)
    )
    db.session.add(history_entry)
    db.session.commit()

    save_user_summary(username, average_score, squat_count)

    # 將所有結果和 conversation_id 傳遞給 chat_report_page
    redirect_url = url_for("chat_report_page",
                            username=username,
                            squat_type=squat_type,
                            answer=llm_answer,
                            conversation_id=conversation_id, # 正確地傳遞 conversation_id
                            report=json.dumps(report_table, ensure_ascii=False),
                            video=result_video_path,
                            images=json.dumps(error_image_paths, ensure_ascii=False),
                            score=average_score,
                            count=squat_count)
                            
    return jsonify({'redirect_url': redirect_url})

def save_user_summary(username: str, avg_score: float, squat_count: int, keep: int = 10):
    item = UserSummary(username=username, avg_score=avg_score, squat_count=squat_count)
    db.session.add(item)
    db.session.flush()

    subq = (db.session.query(UserSummary.id)
            .filter(UserSummary.username == username)
            .order_by(UserSummary.created_at.desc())
            .offset(keep))
    stale_ids = [r[0] for r in subq.all()]
    if stale_ids:
        db.session.query(UserSummary).filter(UserSummary.id.in_(stale_ids)).delete(synchronize_session=False)

    db.session.commit()

# 鏡頭即時分析首頁跳轉
@app.route('/realtime')
def realtime_page():
    username = request.args.get('username')
    squat_type = request.args.get('squat_type')
    return render_template('realtime.html', username=username, squat_type=squat_type)

# 串流：即時鏡頭畫面
@app.route('/video_feed')
def video_feed():
    squat_type = request.args.get('squat_type')
    return Response(gen_video_stream(squat_type),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# 串流：分析資料流 (SSE)
@app.route('/analysis_stream')
def analysis_stream():
    def event_stream():
        while True:
            table = get_latest_analysis_info()
            yield f"data:{json.dumps(table, ensure_ascii=False)}\n\n"
            time.sleep(0.2)
    return Response(event_stream(), mimetype="text/event-stream")

# 儲存即時分析結果
@app.route('/save_realtime_result', methods=['POST'])
def save_realtime_result():
    data = request.json
    username = data['username']
    squat_type = data['squat_type']
    report_table = data['report_table']
    avg_score = data.get('average_score', 0.0)
    squat_count = data.get('squat_count', 0)
    history_entry = AnalysisHistory(
        username=username,
        squat_type=squat_type,
        report_json=json.dumps({
            "report_table": report_table,
            "average_score": avg_score,
            "squat_count": squat_count
        }, ensure_ascii=False)
    )
    db.session.add(history_entry)
    db.session.commit()
    return jsonify({'status': 'ok'})
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    user_message = data.get('message')
    user_id = data.get('user_id')
    conversation_id = data.get('conversation_id')

    payload = {
        "inputs": {},
        "query": user_message,
        "response_mode": "blocking",
        "conversation_id": conversation_id, # 傳遞對話 ID
        "user": user_id
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        dify_response = response.json()
        
        return jsonify({
            "answer": dify_response["answer"],
            "conversation_id": dify_response["conversation_id"]
        })
    except requests.exceptions.RequestException as e:
        print(f"Error calling Dify API: {e}")
        return jsonify({"error": "無法從 AI 取得回覆"}), 500


@app.route('/chat_report')
def chat_report_page():
    # 這裡從 URL 參數接收 Dify 的第一條回覆和對話 ID
    username = request.args.get('username')
    squat_type = request.args.get('squat_type')
    llm_answer = request.args.get('answer') # 接收 Dify 的回覆
    conversation_id = request.args.get('conversation_id') # 接收對話 ID
    
    report = request.args.get('report')
    video = request.args.get('video')
    images = request.args.get('images')
    score = request.args.get('score')
    count = request.args.get('count')

    return render_template("chat_report.html",
        username=username,
        squat_type=squat_type,
        llm_answer=llm_answer,
        conversation_id=conversation_id,
        report_json=report,
        video_path=video,
        error_images_json=images,
        average_score=score,
        squat_count=count
    )

# 歷史紀錄
@app.route('/history')
def history():
    username = request.args.get('username')
    sort = request.args.get('sort', 'timestamp')
    query = AnalysisHistory.query
    if username:
        query = query.filter_by(username=username)
    if sort == 'username':
        query = query.order_by(AnalysisHistory.username)
    else:
        query = query.order_by(AnalysisHistory.timestamp.desc())
    records = query.all()
    return render_template('history.html', records=records)

@app.route('/history/delete/<int:record_id>', methods=['POST'])
def delete_history(record_id):
    record = AnalysisHistory.query.get_or_404(record_id)
    db.session.delete(record)
    db.session.commit()
    return redirect(url_for('history'))

@app.route("/api/summary", methods=["POST"])
def api_write_summary():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    username = (data or {}).get("username")
    avg = (data or {}).get("avg_score")
    cnt = (data or {}).get("squat_count")

    if not username:
        return jsonify({"error": "missing username"}), 400
    try:
        avg = float(avg)
        cnt = int(cnt)
    except Exception:
        return jsonify({"error": "avg_score must be float and squat_count must be int"}), 400

    try:
        save_user_summary(username, avg, cnt, keep=10)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True}), 201

@app.route("/api/summary", methods=["GET"])
def api_read_summary():
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "missing username"}), 400

    rows = (UserSummary.query
            .filter_by(username=username)
            .order_by(UserSummary.created_at.desc(), UserSummary.id.desc())
            .limit(10).all())
    rows = list(reversed(rows))

    data = [{
        "id": r.id,
        "t": r.created_at.isoformat() if r.created_at else None,
        "avg": r.avg_score,
        "cnt": r.squat_count
    } for r in rows]

    return jsonify({"username": username, "data": data}), 200

@app.route("/summary")
def summary_page():
    return render_template("summary_last10.html")

@app.route('/api/summary/<int:sum_id>', methods=['DELETE'])
def api_delete_summary_one(sum_id):
    try:
        row = UserSummary.query.get_or_404(sum_id)
        db.session.delete(row)
        db.session.commit()
        return jsonify({"ok": True}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/health")
def health():
    _load_model_once()
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "num_classes": int(ckpt_meta["num_classes"]) if (ckpt_meta is not None) else None
    })

@app.post("/predict")
def predict():
    _load_model_once()
    if model is None:
        return jsonify({"error": "model not loaded"}), 503

    payload = request.get_json(force=True)

    # 先檢查類別欄位是否缺漏
    missing_categ = [c for c in ckpt_meta["categ_cols"] if c not in payload or str(payload.get(c, "")).strip() == ""]
    # 連續欄位：做轉型並檢查
    invalid_cont = []
    for c in ckpt_meta["cont_cols"]:
        v = _to_float_or_none(payload.get(c))
        if v is None:
            invalid_cont.append(c)
        else:
            payload[c] = v  # 直接覆寫為 float

    
    if "target_ratio" in ckpt_meta["cont_cols"]:
        if (payload.get("target_ratio") is None) and (payload.get("target_weight") is not None) and (payload.get("current_weight") not in (None, 0.0)):
            payload["target_ratio"] = float(payload["target_weight"]) / float(payload["current_weight"])

    if missing_categ or invalid_cont:
        return jsonify({
            "error": "missing/invalid fields",
            "missing_categorical": missing_categ,
            "invalid_continuous": invalid_cont
        }), 400
    with torch.no_grad():
        xc = _encode_categorical(payload).to(DEVICE)
        xn = _scale_continuous(payload).to(DEVICE)
        logits = model(xc, xn)
        probs = _softmax(logits)
        pred_class = int(torch.argmax(logits, dim=1).item())   # 0..K-1
        pred_level = pred_class + 1                             # 回傳 1..K

    return jsonify({"pred_level": pred_level, "probs": probs.tolist()})

# === 測試頁：/predict_demo ===
@app.get("/predict_demo")
def predict_demo():
    _load_model_once()
    options = {}
    categ_cols = []
    cont_cols = []
    if ckpt_meta:
        categ_cols = ckpt_meta.get("categ_cols", [])
        cont_cols  = ckpt_meta.get("cont_cols", [])
        for col in categ_cols:
            try:
                opts = sorted(list(ckpt_meta["cat_maps"][col].keys()))
            except Exception:
                opts = []
            options[col] = opts
    return render_template("predict_demo.html",
                           options=options,
                           categ_cols=categ_cols,
                           cont_cols=cont_cols)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # 啟動前就嘗試載入一次（若沒檔案不會中斷）
    _load_model_once()
    app.run(debug=True)
