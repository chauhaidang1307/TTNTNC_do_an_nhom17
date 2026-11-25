import argparse
import time

import gradio as gr

from engine.board import COLS
from ui.session import (HUMAN, LEVEL_ALPHABETA, LEVEL_HEURISTIC, LEVEL_MCTS, LEVEL_RL, GameSession, default_strength,
                        describe, level_names, strength_choices, strength_is_selectable)

FIRST_CHOICES = ["Người đi trước (Đỏ)", "Máy đi trước (Đỏ)"]
EMPTY_BOARD_HTML = "<p style='text-align:center;padding:2em'>Nhấn <b>Ván mới</b> để bắt đầu.</p>"
DROP_SECONDS = 0.7  # chờ hoạt ảnh quân của người rơi xong rồi máy mới đi (để hoạt ảnh không bị cắt)

# JavaScript của bàn cờ (chạy trong trình duyệt). `element` là phần tử gốc của gr.HTML; `trigger` gửi sự kiện về Python;
# `watch` chạy lại sau mỗi lần bàn cờ được vẽ lại. Quân bóng (.c4-ghost) và dải sáng (.c4-band) chỉ có khi đến lượt người,
# nên lúc máy đang nghĩ thì chuột và phím đều bị bỏ qua.
BOARD_JS = """
const COLS = 7, CELL = 64;
let lastCol = null;
const svg = () => element.querySelector('svg');
const ghost = () => { const s = svg(); return s ? s.querySelector('.c4-ghost') : null; };
function applyCol(c) {
  const s = svg(); if (!s) return;
  const g = s.querySelector('.c4-ghost'), b = s.querySelector('.c4-band');
  if (g) g.setAttribute('cx', c * CELL + CELL / 2);
  if (b) b.setAttribute('x', c * CELL);
}
function colFromMouse(e) {
  const r = svg().getBoundingClientRect();
  return Math.max(0, Math.min(COLS - 1, Math.floor((e.clientX - r.left) / r.width * COLS)));
}
function drop(c) { lastCol = c; applyCol(c); trigger('click', {col: c}); }
element.addEventListener('mousemove', (e) => { if (!ghost() || !svg()) return; lastCol = colFromMouse(e); applyCol(lastCol); });
element.addEventListener('click', (e) => { if (!ghost() || !svg()) return; drop(colFromMouse(e)); });
watch('value', () => { if (lastCol !== null) applyCol(lastCol); });
// Có nhiều bàn cờ có tương tác (Người-Máy và Người-Người): mỗi bàn tự đăng ký một hàm thả quân; phím chỉ thả vào
// bàn đang HIỂN THỊ (tab đang mở) và đang có quân bóng. Bàn đã bị gỡ khỏi trang có offsetParent null nên bị bỏ qua.
window.__c4boards = window.__c4boards || [];
window.__c4boards.push((c) => { if (element.offsetParent !== null && ghost()) { drop(c); return true; } return false; });
if (!window.__c4keys) {
  window.__c4keys = true;
  document.addEventListener('keydown', (e) => {
    const t = e.target, tag = t && t.tagName ? t.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select' || (t && t.isContentEditable)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key >= '1' && e.key <= '7') {
      const c = parseInt(e.key, 10) - 1;
      for (const f of window.__c4boards) { if (f(c)) break; }
    }
  });
}
"""

CSS = """
/* Độ rộng khung ứng dụng CỐ ĐỊNH (100%, tối đa 1100 px). Nếu chỉ có max-width và margin auto thì khung co theo độ dài chữ
   bên phải và bàn cờ "phóng to thu nhỏ" sau mỗi nước (đã đo: khung 829 px và 938 px luân phiên). */
.gradio-container {width: 100% !important; max-width: 1100px !important; margin-left: auto !important; margin-right: auto !important;}
#board-area, #board-area3 {cursor: pointer; user-select: none;}
/* Khung chứa bàn cờ có độ rộng xác định (100% cột). Nếu để tự do, khung co giãn theo nội dung và bàn cờ "phóng to thu nhỏ"
   qua lại sau mỗi lần vẽ lại (đã đo: 426 px và 448 px luân phiên mỗi nước). */
#board-area, #board-area2, #board-area3, #board-area .html-container, #board-area2 .html-container,
#board-area3 .html-container, #board-area .prose, #board-area2 .prose, #board-area3 .prose
{width: 100% !important; max-width: none !important; box-sizing: border-box;}
/* Cố định tỉ lệ 3:2 của hai cột, không phụ thuộc độ dài chữ bên phải (nếu không bàn cờ co giãn theo từng nước đi) */
.c4-left {flex: 3 1 0% !important; min-width: 0 !important;}
.c4-right {flex: 2 1 0% !important; min-width: 0 !important; overflow-wrap: anywhere;}
"""


def _view(session, animate=False):
    # Bộ đầu ra chung: (session, bàn cờ, trạng thái, thông tin nước vừa đi)
    return session, session.render(animate=animate), session.status_text(), session.last_move_text()


def _no_game():
    return None, EMPTY_BOARD_HTML, "Chưa có ván. Nhấn **Ván mới**.", ""


# ---------------- Người – Máy ----------------

def new_game_hvm(ai_spec, first, seed):
    human_first = first == FIRST_CHOICES[0]
    seed = int(seed or 0)
    session = GameSession(HUMAN, ai_spec, seed) if human_first else GameSession(ai_spec, HUMAN, seed)
    ai_moved = False
    if session.current_is_ai():
        session.step_ai()
        ai_moved = True
    return _view(session, animate=ai_moved)


def drop_piece(session, evt: gr.EventData):
    # evt.col: cột do JavaScript gửi lên (click chuột hoặc phím 1 đến 7).
    if session is None:
        return _no_game()
    col = int(evt.col)
    session.hover_col = max(0, min(COLS - 1, col))
    try:
        session.play_human(col)
    except ValueError as err:
        s, board_html, _, info = _view(session)
        return s, board_html, f"⚠️ {err} {session.status_text()}", info
    return _view(session, animate=True)


def ai_reply(session):
    # Chạy sau drop_piece (nối bằng .then): chờ hoạt ảnh quân của người xong rồi mới cho máy đi.
    if session is None:
        return _no_game()
    if session.current_is_ai():
        if session.moves and session.moves[-1]["human"]:
            time.sleep(DROP_SECONDS)
        session.step_ai()
        return _view(session, animate=True)
    return _view(session)


def undo(session):
    if session is None:
        return _no_game()
    session.undo_turn()
    ai_moved = False
    if session.current_is_ai():  # máy đi trước và bị hoàn tác hết: cho máy đi lại nước đầu
        session.step_ai()
        ai_moved = True
    return _view(session, animate=ai_moved)


# ---------------- Người – Người ----------------

def new_game_hvh():
    # Hai người chơi trên cùng một máy; dùng lại drop_piece và undo của chế độ Người – Máy (không có máy đáp lại).
    return _view(GameSession(HUMAN, HUMAN))


# ---------------- Máy – Máy ----------------

def new_game_mvm(spec1, spec2, seed):
    return _view(GameSession(spec1, spec2, int(seed or 0)))


def mvm_step(session):
    if session is None:
        return _no_game()
    if not session.is_over():
        session.step_ai()
    return _view(session, animate=True)


def mvm_run(session, delay):
    # Hàm sinh (generator): trả về từng nước để màn hình cập nhật dần.
    if session is None:
        yield _no_game()
        return
    while not session.is_over():
        session.step_ai()
        yield _view(session, animate=True)
        time.sleep(max(float(delay), 0.75))  # đủ cho hoạt ảnh rơi
    yield _view(session)


# ---------------- Chọn đối thủ theo cấp rồi theo độ mạnh ----------------

def level_changed(level):
    # Đổi cấp: cập nhật ô độ mạnh (chỉ hiện khi cấp đó có nhiều mức) và dòng mô tả.
    spec = default_strength(level)
    label = "Phiên bản mô hình" if level == LEVEL_RL else "Độ mạnh"
    dd = gr.Dropdown(choices=strength_choices(level), value=spec, label=label, visible=strength_is_selectable(level))
    return dd, describe(level, spec)


def strength_changed(level, spec):
    return describe(level, spec)


def ai_selector(title, default_level, default_spec):
    # Hai ô nối nhau: cấp (theo bảng cấp của đề) và độ mạnh, kèm một dòng mô tả. Trả về (ô cấp, ô độ mạnh).
    level = gr.Dropdown(choices=level_names(), value=default_level, label=title)
    strength = gr.Dropdown(choices=strength_choices(default_level), value=default_spec, label="Độ mạnh",
                           visible=strength_is_selectable(default_level))
    desc = gr.Markdown(describe(default_level, default_spec))
    level.change(level_changed, [level], [strength, desc], show_progress="hidden")
    strength.change(strength_changed, [level, strength], [desc], show_progress="hidden")
    return level, strength


# ---------------- Dựng giao diện ----------------

def build_app():
    with gr.Blocks(title="Connect Four AI") as demo:
        gr.Markdown("# Connect Four AI\n"
                    "Đồ án tác tử đa cấp độ cho Connect Four (7 cột x 6 hàng). Đỏ đi trước; cột đánh số 1 đến 7.")
        with gr.Tab("Người - Máy"):
            state = gr.State(None)
            with gr.Row():
                with gr.Column(scale=3, elem_classes=["c4-left"]):
                    board_html = gr.HTML(EMPTY_BOARD_HTML, elem_id="board-area", js_on_load=BOARD_JS)
                    gr.Markdown("🖱️ **Di chuột** trên bàn cờ để chọn cột, **click** để thả quân. "
                                "Hoặc nhấn phím **1 đến 7** (cột 1 đến 7).")
                with gr.Column(scale=2, elem_classes=["c4-right"]):
                    _, ai_dd = ai_selector("Đối thủ AI", LEVEL_HEURISTIC, "heuristic")
                    gr.Markdown("*Đổi đối thủ xong, nhấn **Ván mới** để áp dụng.*")
                    first_rd = gr.Radio(FIRST_CHOICES, value=FIRST_CHOICES[0], label="Lượt đi")
                    seed_n = gr.Number(value=0, precision=0, label="Seed (tái lập ván)")
                    with gr.Row():
                        new_btn = gr.Button("Ván mới", variant="primary")
                        undo_btn = gr.Button("Hoàn tác")
                    status = gr.Markdown("")
                    info = gr.Markdown("")
            outs = [state, board_html, status, info]

            # show_progress="hidden": không phủ lớp mờ "processing" lên bàn cờ khi máy đang nghĩ (làm mờ hoạt ảnh)
            new_btn.click(new_game_hvm, [ai_dd, first_rd, seed_n], outs, api_name="new_game", show_progress="hidden")
            board_html.click(drop_piece, [state], outs, api_name="drop", show_progress="hidden") \
                .then(ai_reply, [state], outs, api_name="ai_reply", show_progress="hidden")
            undo_btn.click(undo, [state], outs, api_name="undo", show_progress="hidden")
            demo.load(new_game_hvm, [ai_dd, first_rd, seed_n], outs)  # tự bắt đầu một ván khi mở trang

        with gr.Tab("Người - Người"):
            state3 = gr.State(None)
            with gr.Row():
                with gr.Column(scale=3, elem_classes=["c4-left"]):
                    board3 = gr.HTML(EMPTY_BOARD_HTML, elem_id="board-area3", js_on_load=BOARD_JS)
                    gr.Markdown("Hai người **lần lượt** di chuột và click (hoặc nhấn phím **1 đến 7**) trên cùng một máy. "
                                "Đỏ đi trước.")
                with gr.Column(scale=2, elem_classes=["c4-right"]):
                    with gr.Row():
                        new3 = gr.Button("Ván mới", variant="primary")
                        undo3 = gr.Button("Hoàn tác")
                    status3 = gr.Markdown("")
                    info3 = gr.Markdown("")
            outs3 = [state3, board3, status3, info3]
            new3.click(new_game_hvh, None, outs3, api_name="new_game_hvh", show_progress="hidden")
            board3.click(drop_piece, [state3], outs3, api_name="drop_hvh", show_progress="hidden")
            undo3.click(undo, [state3], outs3, api_name="undo_hvh", show_progress="hidden")
            demo.load(new_game_hvh, None, outs3)  # tự bắt đầu một ván khi mở trang

        with gr.Tab("Máy - Máy"):
            state2 = gr.State(None)
            with gr.Row():
                with gr.Column(scale=3, elem_classes=["c4-left"]):
                    board2 = gr.HTML(EMPTY_BOARD_HTML, elem_id="board-area2", js_on_load=None)
                    with gr.Row():
                        step2 = gr.Button("Nước kế tiếp")
                        run2 = gr.Button("Chạy cả ván", variant="primary")
                with gr.Column(scale=2, elem_classes=["c4-right"]):
                    _, p1_dd = ai_selector("Đỏ (đi trước)", LEVEL_ALPHABETA, "ab-d4")
                    _, p2_dd = ai_selector("Vàng (đi sau)", LEVEL_MCTS, "mcts-s1000")
                    seed2 = gr.Number(value=0, precision=0, label="Seed")
                    delay = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="Nghỉ giữa các nước (giây, tối thiểu 0,75)")
                    new2 = gr.Button("Ván mới")
                    status2 = gr.Markdown("")
                    info2 = gr.Markdown("")
            outs2 = [state2, board2, status2, info2]
            new2.click(new_game_mvm, [p1_dd, p2_dd, seed2], outs2, api_name="mvm_new", show_progress="hidden")
            step2.click(mvm_step, [state2], outs2, api_name="mvm_step", show_progress="hidden")
            run2.click(mvm_run, [state2, delay], outs2, api_name="mvm_run", show_progress="hidden")
    return demo


def main():
    ap = argparse.ArgumentParser(description="Giao diện web Connect Four (Gradio)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true", help="không tự mở trình duyệt")
    args = ap.parse_args()
    # Gradio 6: theme và css truyền vào launch() (không còn ở Blocks).
    build_app().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser,
                       theme=gr.themes.Soft(), css=CSS)


if __name__ == "__main__":
    main()
