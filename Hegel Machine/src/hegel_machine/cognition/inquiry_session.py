"""Question-local session state shared by the world and numerical researchers."""

from copy import deepcopy


def begin_session(sessions, stack, question, *, start_calls, calls, total_calls,
                  session_id=None, assignment=None):
    session = sessions.setdefault(session_id or question['id'], {
        'question_id': question['id'], 'notes': '', 'events': [],
        'invocations': 0, 'model_calls': 0, 'status': 'open'})
    session['question_id'] = question['id']
    session['invocations'] += 1
    frame = {'question_id': question['id'], 'question': question,
             'assignment': deepcopy(question), 'start_calls': start_calls,
             'call_limit': min(start_calls + calls, total_calls,
                               *(item['call_limit'] for item in stack)),
             'stop_requested': False}
    if assignment is not None:
        frame['assignment']['question'] = assignment
    return session, frame


def enter_notebook(workspace, session, frame):
    path = workspace / 'research_notes.md'
    frame['parent_notes'] = path.read_text() if path.exists() else None
    path.write_text(session['notes'])


def leave_notebook(workspace, session, frame, status):
    path = workspace / 'research_notes.md'
    if frame['parent_notes'] is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(frame['parent_notes'])
    session['status'] = status
