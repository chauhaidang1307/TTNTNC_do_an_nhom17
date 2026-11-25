import re
from types import SimpleNamespace

import pytest
from engine import board
from engine.board import COLS, PLAYER_1, PLAYER_2
from rl.checkpoint import save_checkpoint
from rl.network import QNet
from ui import gui
from arena.tournament import make_agent
from ui import session as S
from ui.session import HUMAN, GameSession, dqn_choices, render_svg, winning_cells


def play(moves):
    b = board.Board()
    for col in moves:
        b.step(col)
    return b


# ---------- ô thắng ----------

def test_winning_cells_horizontal_vertical_diagonal():
    assert sorted(winning_cells(play([0, 0, 1, 1, 2, 2, 3]))) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert sorted(winning_cells(play([0, 1, 0, 1, 0, 1, 0]))) == [(0, 0), (1, 0), (2, 0), (3, 0)]
    diag = play([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3])  # chéo (0,0),(1,1),(2,2),(3,3) cho người 1
    assert sorted(winning_cells(diag)) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_no_winning_cells_when_no_winner():
    assert winning_cells(play([3, 3, 2])) == []


# ---------- vẽ bàn cờ ----------

def test_render_svg_empty_board_has_frame_with_42_holes_and_no_pieces():
    svg = render_svg(board.Board())
    assert svg.startswith("<svg") and 'fill-rule="evenodd"' in svg
    assert svg.count("<circle") == 0  # bàn trống: không có quân; 42 lỗ nằm trong một đường vẽ của khung
    frame = re.search(r'<path d="([^"]+)"', svg).group(1)
    assert frame.count("a") >= 2 * 42  # mỗi lỗ là hai cung tròn


def test_render_svg_draws_one_piece_per_move_with_letters():
    svg = render_svg(play([3, 3, 2]))
    assert len(re.findall(r">X<", svg)) == 2 and len(re.findall(r">O<", svg)) == 1


def test_render_svg_highlights_four_winning_cells_and_marks_last_move():
    svg = render_svg(play([0, 0, 1, 1, 2, 2, 3]))
    assert svg.count('stroke="#111" stroke-width="5"') == 4  # vòng viền bốn ô thắng
    assert svg.count('r="4" fill="#111"') == 1  # chấm đánh dấu quân vừa đi


def test_animation_only_on_last_piece_and_only_when_requested():
    b = play([3, 3, 2])
    assert "animateTransform" not in render_svg(b)
    svg = render_svg(b, animate=True)
    assert svg.count("animateTransform") == 1
    assert 'r="4" fill="#111"' not in svg  # đang rơi thì chưa hiện chấm đánh dấu
    # quân vừa đi ở hàng 0 của cột 2: rơi từ hàng xem trước xuống, độ dịch = (ROWS - 0) * CELL
    assert f'from="0 -{board.ROWS * 64}"' in svg


def test_falling_piece_is_drawn_before_frame_so_it_goes_behind_it():
    svg = render_svg(play([3]), animate=True)
    assert svg.index("animateTransform") < svg.index('fill-rule="evenodd"')


def test_preview_ghost_and_column_band():
    b = play([3])
    with_preview = render_svg(b, preview_col=5, preview_player=PLAYER_2)
    assert 'stroke-dasharray="6 4"' in with_preview and 'fill-opacity="0.12"' in with_preview
    assert 'stroke-dasharray="6 4"' not in render_svg(b)
    assert 'stroke-dasharray="6 4"' not in render_svg(play([0, 0, 1, 1, 2, 2, 3]), preview_col=5, preview_player=PLAYER_2)


def test_preview_column_is_clamped():
    assert "<svg" in render_svg(board.Board(), preview_col=99, preview_player=PLAYER_1)
    assert "<svg" in render_svg(board.Board(), preview_col=-3, preview_player=PLAYER_1)


def test_session_render_shows_preview_only_on_human_turn():
    s = GameSession(HUMAN, "random", seed=0)
    assert 'stroke-dasharray="6 4"' in s.render(preview_col=2)
    s.play_human(3)  # đến lượt máy
    assert 'stroke-dasharray="6 4"' not in s.render(preview_col=2)


# ---------- phiên chơi người–máy ----------

def test_human_move_and_ai_reply():
    s = GameSession(HUMAN, "random", seed=1)
    assert s.current_is_human()
    s.play_human(3)
    assert s.current_is_ai() and s.board.grid[0][3] == PLAYER_1
    rec = s.step_ai()
    assert rec["human"] is False and len(s.moves) == 2 and s.current_is_human()
    assert rec["time_ms"] >= 0


def test_human_cannot_play_illegal_or_out_of_turn():
    s = GameSession(HUMAN, "random", seed=1)
    for _ in range(3):
        s.play_human(0)
        s.step_ai()
    with pytest.raises(ValueError):
        s.play_human(99)
    s2 = GameSession(HUMAN, "random")
    s2.play_human(0)
    with pytest.raises(ValueError):
        s2.play_human(1)  # đang đến lượt máy


def test_full_column_rejected():
    s = GameSession(HUMAN, "random", seed=0)
    s.board = play([3] * 6)
    s.board.to_move = PLAYER_1
    with pytest.raises(ValueError):
        s.play_human(3)


def test_ai_first_session_and_step_ai_refuses_human_turn():
    s = GameSession("random", HUMAN, seed=0)
    assert s.current_is_ai()
    s.step_ai()
    assert s.current_is_human()
    with pytest.raises(ValueError):
        s.step_ai()


def test_undo_turn_removes_human_move_and_following_ai_moves():
    s = GameSession(HUMAN, "random", seed=0)
    s.play_human(2)
    s.step_ai()
    s.play_human(4)
    s.step_ai()
    assert s.undo_turn() and len(s.moves) == 2 and s.board.moves_made == 2
    assert s.undo_turn() and len(s.moves) == 0 and s.board.moves_made == 0
    assert s.undo_turn() is False


def test_undo_does_nothing_if_human_has_not_moved():
    s = GameSession("random", HUMAN, seed=0)
    s.step_ai()
    assert s.undo_turn() is False and len(s.moves) == 1


def test_status_texts():
    s = GameSession(HUMAN, "random", seed=0)
    assert "Lượt của bạn" in s.status_text()
    s.play_human(0)
    assert "Lượt của máy" in s.status_text()
    s.board = play([0, 0, 1, 1, 2, 2, 3])
    assert "thắng" in s.status_text() and "Đỏ" in s.status_text()


def test_last_move_text_shows_cost_of_ai():
    s = GameSession(HUMAN, "ab-d3", seed=0)
    s.play_human(3)
    s.step_ai()
    txt = s.last_move_text()
    assert "ms" in txt and "nút" in txt and "độ sâu 3" in txt
    m = GameSession(HUMAN, "mcts-s50", seed=0)
    m.play_human(3)
    m.step_ai()
    assert "mô phỏng" in m.last_move_text()


# ---------- máy–máy ----------

def test_ai_vs_ai_plays_a_complete_legal_game():
    s = GameSession("ab-d2", "random", seed=3)
    while not s.is_over():
        s.step_ai()
    assert s.board.is_terminal() and len(s.moves) == s.board.moves_made
    assert all(not m["human"] for m in s.moves)


# ---------- chọn đối thủ theo cấp và độ mạnh ----------

def test_levels_follow_the_assignment_table(tmp_path):
    assert S.level_names(tmp_path / "none") == [S.LEVEL_RANDOM, S.LEVEL_HEURISTIC, S.LEVEL_ALPHABETA, S.LEVEL_MCTS]


def test_rl_level_appears_only_when_a_checkpoint_exists(tmp_path):
    assert S.LEVEL_RL not in S.level_names(tmp_path)
    save_checkpoint(tmp_path / "RL-X" / "last.pt", QNet(4, 8), {"channels": 4, "hidden": 8}, 1)
    assert S.level_names(tmp_path)[-1] == S.LEVEL_RL


def test_every_strength_value_is_a_buildable_agent():
    for level in (S.LEVEL_RANDOM, S.LEVEL_HEURISTIC, S.LEVEL_ALPHABETA, S.LEVEL_MCTS):
        for _label, spec in S.strength_choices(level):
            assert make_agent(spec, 0) is not None


def test_strength_selector_visible_only_for_levels_with_several_modes():
    assert not S.strength_is_selectable(S.LEVEL_RANDOM)
    assert not S.strength_is_selectable(S.LEVEL_HEURISTIC)
    assert S.strength_is_selectable(S.LEVEL_ALPHABETA) and S.strength_is_selectable(S.LEVEL_MCTS)


def test_strength_labels_have_word_and_the_real_parameter():
    # Chữ Dễ/Vừa/Khó kèm thông số thật trong ngoặc; thông số trong nhãn phải khớp tên tác tử phía sau.
    for level, prefix, unit in ((S.LEVEL_ALPHABETA, "ab-d", "độ sâu"), (S.LEVEL_MCTS, "mcts-s", "mô phỏng")):
        choices = S.strength_choices(level)
        assert [label.split(" ")[0] for label, _ in choices] == ["Dễ", "Vừa", "Khó"]
        for label, spec in choices:
            number = spec[len(prefix):]
            assert f"({unit} {number})" in label or f"({number} {unit})" in label


def test_strength_order_alphabeta_depth_and_mcts_budget_increase():
    depths = [int(spec[4:]) for _, spec in S.strength_choices(S.LEVEL_ALPHABETA)]
    sims = [int(spec[6:]) for _, spec in S.strength_choices(S.LEVEL_MCTS)]
    assert depths == sorted(depths) and sims == sorted(sims)


def test_default_strengths():
    assert S.default_strength(S.LEVEL_ALPHABETA) == "ab-d4"
    assert S.default_strength(S.LEVEL_MCTS) == "mcts-s1000"
    assert S.default_strength(S.LEVEL_HEURISTIC) == "heuristic"


def test_rl_default_is_last_checkpoint_of_newest_run(tmp_path):
    for run in ("RL-001", "RL-002"):
        for ck in ("best", "last"):
            save_checkpoint(tmp_path / run / f"{ck}.pt", QNet(4, 8), {"channels": 4, "hidden": 8}, 1)
    assert S.default_strength(S.LEVEL_RL, tmp_path) == "dqn_RL-002_last"
    labels = [label for label, _ in S.strength_choices(S.LEVEL_RL, tmp_path)]
    assert "RL-002 – bản cuối" in labels


def test_describe_puts_technical_details_in_the_description_not_the_labels():
    assert "tìm trước 4 nước" in S.describe(S.LEVEL_ALPHABETA, "ab-d4")
    assert "1000 ván thử" in S.describe(S.LEVEL_MCTS, "mcts-s1000")
    assert "Alpha-Beta" in S.describe(S.LEVEL_ALPHABETA, "ab-d2")


def test_dqn_choices_scan_checkpoints(tmp_path):
    assert dqn_choices(tmp_path / "none") == []
    save_checkpoint(tmp_path / "RL-X" / "best.pt", QNet(4, 8), {"channels": 4, "hidden": 8}, 1)
    assert dqn_choices(tmp_path) == [("RL-X – bản tốt nhất trên ván dev", "dqn_RL-X_best")]


def test_unknown_agent_spec_raises():
    with pytest.raises(ValueError):
        GameSession(HUMAN, "khong-co")


# ---------- lớp giao diện (gọi hàm trực tiếp, không cần trình duyệt) ----------

@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(gui.time, "sleep", lambda _s: None)  # bỏ thời gian chờ hoạt ảnh trong test


def click(col):
    # Giả lập dữ liệu sự kiện mà JavaScript gửi lên (trigger('click', {col: ...})).
    return SimpleNamespace(col=col)


def test_gui_new_game_human_first_and_ai_first():
    s, html, status, info = gui.new_game_hvm("random", gui.FIRST_CHOICES[0], 0)
    assert s.current_is_human() and "<svg" in html and "Lượt của bạn" in status
    assert 'class="c4-ghost"' in html and 'class="c4-band"' in html  # JS cần hai phần tử này để bám theo chuột
    s2, html2, _, info2 = gui.new_game_hvm("random", gui.FIRST_CHOICES[1], 0)
    assert len(s2.moves) == 1 and "đi cột" in info2
    assert "animateTransform" in html2  # nước đi đầu của máy có hoạt ảnh rơi
    assert 'class="c4-ghost"' not in html2 or s2.current_is_human()


def test_gui_drop_then_ai_reply_flow_and_animation():
    s, *_ = gui.new_game_hvm("random", gui.FIRST_CHOICES[0], 0)
    s, html, status, _ = gui.drop_piece(s, click(3))
    assert len(s.moves) == 1 and "Lượt của máy" in status and "animateTransform" in html
    assert 'class="c4-ghost"' not in html  # máy đang nghĩ: không có quân bóng nên chuột/phím bị JS bỏ qua
    s, html, status, _ = gui.ai_reply(s)
    assert len(s.moves) == 2 and "Lượt của bạn" in status and "animateTransform" in html
    assert 'class="c4-ghost"' in html


def test_gui_clicked_column_is_what_gets_dropped_and_ghost_stays_there():
    s, *_ = gui.new_game_hvm("random", gui.FIRST_CHOICES[0], 0)
    s, *_ = gui.drop_piece(s, click(5))
    assert s.board.grid[0][5] == PLAYER_1 and s.hover_col == 5
    s, html, *_ = gui.ai_reply(s)
    assert re.search(rf'class="c4-ghost" cx="{5 * 64 + 32}(\.0)?"', html)  # quân bóng ở lại cột vừa chọn sau khi vẽ lại


def test_gui_click_column_is_clamped_and_illegal_is_rejected():
    s, *_ = gui.new_game_hvm("random", gui.FIRST_CHOICES[0], 0)
    s2, _, status, _ = gui.drop_piece(s, click(99))
    assert s2 is s and "⚠️" in status and s.hover_col == 6


def test_gui_full_column_click_keeps_state_and_warns():
    s, *_ = gui.new_game_hvm("random", gui.FIRST_CHOICES[0], 0)
    s.board = play([3] * 6)
    s.board.to_move = PLAYER_1
    s2, _, status, _ = gui.drop_piece(s, click(3))
    assert s2 is s and "⚠️" in status


def test_gui_undo_has_no_animation():
    s, *_ = gui.new_game_hvm("random", gui.FIRST_CHOICES[0], 0)
    s, *_ = gui.drop_piece(s, click(3))
    s, *_ = gui.ai_reply(s)
    s, html, _, _ = gui.undo(s)
    assert len(s.moves) == 0 and "animateTransform" not in html


def test_gui_handles_no_session():
    assert gui.drop_piece(None, click(3))[0] is None
    assert gui.ai_reply(None)[0] is None
    assert gui.undo(None)[0] is None
    assert gui.mvm_step(None)[0] is None


def test_gui_mvm_run_generator_finishes_game():
    s, *_ = gui.new_game_mvm("random", "random", 1)
    frames = list(gui.mvm_run(s, 0))
    assert s.is_over() and frames[-1][2].startswith("**Kết thúc")


def test_build_app_creates_blocks():
    assert gui.build_app() is not None


def test_level_changed_updates_strength_box_and_description():
    dd, text = gui.level_changed(S.LEVEL_ALPHABETA)
    assert dd.visible is True and dd.value == "ab-d4" and "Alpha-Beta" in text
    dd, text = gui.level_changed(S.LEVEL_RANDOM)
    assert dd.visible is False and dd.value == "random"


def test_strength_changed_returns_description():
    assert "tìm trước 6 nước" in gui.strength_changed(S.LEVEL_ALPHABETA, "ab-d6")


def test_new_game_works_for_every_strength_choice():
    for level in (S.LEVEL_RANDOM, S.LEVEL_HEURISTIC, S.LEVEL_ALPHABETA, S.LEVEL_MCTS):
        for _label, spec in S.strength_choices(level):
            s, html, status, _ = gui.new_game_hvm(spec, gui.FIRST_CHOICES[0], 0)
            assert s.current_is_human() and "<svg" in html


# ---------- đánh số cột hiển thị 1 đến 7 ----------

def test_column_numbers_shown_to_user_are_one_based():
    svg = render_svg(board.Board())
    labels = re.findall(r'fill-opacity="0.7">(\d)</text>', svg)
    assert labels == ["1", "2", "3", "4", "5", "6", "7"]


def test_last_move_text_is_one_based():
    s = GameSession(HUMAN, "random", seed=0)
    s.play_human(0)
    assert "đi cột 1." in s.last_move_text()
    s2 = GameSession(HUMAN, "random", seed=0)
    s2.play_human(6)
    assert "đi cột 7." in s2.last_move_text()


def test_illegal_column_message_is_one_based():
    s = GameSession(HUMAN, "random", seed=0)
    s.board = play([3] * 6)
    s.board.to_move = PLAYER_1
    with pytest.raises(ValueError, match="Cột 4"):
        s.play_human(3)


def test_board_js_has_mouse_click_and_number_keys():
    js = gui.BOARD_JS
    for needle in ("mousemove", "'click'", "keydown", "e.key >= '1' && e.key <= '7'", "trigger('click'", "c4-ghost"):
        assert needle in js


# ---------- Người – Người ----------

def test_human_vs_human_alternates_colors_and_never_asks_the_machine():
    s = GameSession(HUMAN, HUMAN)
    assert s.both_human and s.current_is_human() and not s.current_is_ai()
    s.play_human(3)
    assert s.board.grid[0][3] == PLAYER_1 and s.current_is_human() and s.board.to_move == PLAYER_2
    s.play_human(3)
    assert s.board.grid[1][3] == PLAYER_2 and s.board.to_move == PLAYER_1
    with pytest.raises(ValueError):
        s.step_ai()  # không có tác tử nào để đi hộ


def test_human_vs_human_player_names_and_status():
    s = GameSession(HUMAN, HUMAN)
    assert "Người chơi 1" in s.status_text() and "Đỏ" in s.status_text()
    s.play_human(0)
    assert "Người chơi 2" in s.status_text() and "Vàng" in s.status_text()
    assert "Người chơi 1" in s.last_move_text() and "đi cột 1." in s.last_move_text()


def test_human_vs_human_winner_is_named():
    s = GameSession(HUMAN, HUMAN)
    for col in [0, 0, 1, 1, 2, 2, 3]:  # người 1 thắng hàng ngang
        s.play_human(col)
    assert s.is_over() and "Người chơi 1" in s.status_text() and "thắng" in s.status_text()
    with pytest.raises(ValueError):
        s.play_human(4)


def test_human_vs_human_draw_message():
    from tests.test_mcts import DRAW_GAME
    s = GameSession(HUMAN, HUMAN)
    for col in DRAW_GAME:
        s.play_human(col)
    assert s.is_over() and "hòa" in s.status_text()


def test_human_vs_human_undo_removes_exactly_one_move():
    s = GameSession(HUMAN, HUMAN)
    for col in (2, 4, 2):
        s.play_human(col)
    assert s.undo_turn() and len(s.moves) == 2 and s.board.moves_made == 2 and s.board.to_move == PLAYER_1
    assert s.undo_turn() and s.undo_turn() and len(s.moves) == 0
    assert s.undo_turn() is False


def test_human_vs_human_illegal_move_is_rejected_without_changing_turn():
    s = GameSession(HUMAN, HUMAN)
    for _ in range(3):
        s.play_human(3)
        s.play_human(3)  # cột 4 (chỉ số 3) đầy sau 6 quân
    to_move = s.board.to_move
    with pytest.raises(ValueError, match="Cột 4"):
        s.play_human(3)
    assert s.board.to_move == to_move


def test_human_vs_human_shows_ghost_in_color_of_player_to_move():
    s = GameSession(HUMAN, HUMAN)
    assert "#e53935" in re.search(r'class="c4-ghost"[^>]*>', s.render()).group(0)  # đỏ
    s.play_human(0)
    assert "#fdd835" in re.search(r'class="c4-ghost"[^>]*>', s.render()).group(0)  # vàng


def test_gui_human_vs_human_flow_has_no_ai_reply():
    s, html, status, _ = gui.new_game_hvh()
    assert s.both_human and "Người chơi 1" in status and 'class="c4-ghost"' in html
    s, html, status, _ = gui.drop_piece(s, click(3))
    assert len(s.moves) == 1 and "Người chơi 2" in status and "animateTransform" in html
    assert 'class="c4-ghost"' in html  # vẫn đến lượt người nên còn quân bóng cho người kế tiếp
    s, html, status, _ = gui.drop_piece(s, click(4))
    assert len(s.moves) == 2 and "Người chơi 1" in status
    s, html, status, info = gui.undo(s)
    assert len(s.moves) == 1 and "Người chơi 2" in status and "animateTransform" not in html


def test_board_js_registers_each_board_and_keys_go_to_the_visible_one():
    js = gui.BOARD_JS
    assert "window.__c4boards" in js and "element.offsetParent !== null" in js
    assert "for (const f of window.__c4boards)" in js


def test_app_has_three_tabs_in_order():
    demo = gui.build_app()
    labels = [b.label for b in demo.blocks.values() if type(b).__name__ == "Tab"]
    assert labels == ["Người – Máy", "Người – Người", "Máy – Máy"]
