'''In this exercise you need to put all code together to make the robot be able to stand up by its own.

* Task:
    complete the `StandingUpAgent.standing_up` function, e.g. call keyframe motion corresponds to current posture

'''

from recognize_posture import PostureRecognitionAgent
from keyframes import leftBackToStand, rightBackToStand, leftBellyToStand, rightBellyToStand
from recognize_posture import PostureRecognitionAgent
import time

def _append_stabilize_to_keyframes(names, times_list, keys, hold_time=2.0, forward_delta=0.08):
    """
    Fügt Stabilisations-Keyframes an die bestehenden keyframes an.
    - Erhält die bestehende Struktur der key-Einträge (z.B. [angle, [3,...], [3,...]]).
    - Fügt zwei Zeitpunkte pro Gelenk hinzu: t_last+0.5s (Vorwärtskorrektur) und t_last+hold_time (Hold).
    """
    # defensive copy
    names = list(names)
    times = [list(t) for t in times_list]
    keys2 = [list(k) for k in keys]

    # Indices der relevanten Gelenke (hip/ankle pitch)
    hip_idx = [i for i, n in enumerate(names) if 'HipPitch' in n or ('Hip' in n and 'Pitch' in n)]
    ankle_idx = [i for i, n in enumerate(names) if 'AnklePitch' in n or ('Ankle' in n and 'Pitch' in n)]

    for i in range(len(names)):
        # letzte Zeit / letzter Key
        last_t = times[i][-1] if len(times[i]) > 0 else 0.0
        last_key = keys2[i][-1] if len(keys2[i]) > 0 else 0.0

        # bestimme aktuellen Winkelwert (skalar)
        if isinstance(last_key, list) and len(last_key) > 0:
            last_angle = last_key[0]
        else:
            last_angle = last_key

        # neue Winkel (Vorwärtskorrektur für Hüfte, Kompensation bei Knöchel)
        new_angle = last_angle
        if i in hip_idx:
            new_angle = last_angle - forward_delta
        elif i in ankle_idx:
            new_angle = last_angle + forward_delta

        # Baue neuen Key-Eintrag in derselben Form wie last_key (wenn last_key Liste ist, kopieren wir die Interpolations-Parameter)
        if isinstance(last_key, list) and len(last_key) >= 3:
            # keep interpolation coefficients (zweites und drittes Element), nur Winkel ersetzen
            new_entry = [new_angle, last_key[1], last_key[2]]
        elif isinstance(last_key, list) and len(last_key) == 1:
            new_entry = [new_angle]
        else:
            # last_key war ein Skalar → behalten Skalar-Format
            new_entry = new_angle

        # Zeiten: intermediate (t1) und hold (t2)
        t1 = last_t + 0.5
        t2 = last_t + hold_time

        # füge die zwei neuen Punkte an times und keys2 an
        times[i].append(t1)
        keys2[i].append(new_entry)
        times[i].append(t2)
        keys2[i].append(new_entry)

    return names, times, keys2

class StandingUpAgent(PostureRecognitionAgent):
    # 1) __init__ Ergänzungen
    def __init__(self, simspark_ip='localhost', simspark_port=3100,
                 teamname='DAInamite', player_id=0, sync_mode=True):
        super(StandingUpAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.standing_up_started = False
    
        # internal state for robust stand-up
        self._stand_state = None
        self._max_retries = 3
        self._post_hold_until = 0.0            # keep stiffness for a short time after a seq
        self._force_stiffness_until = 0.0      # keep stiffness on for entire seq+post-hold
        self._stiffness_value = 1.0            # value used for all joints during stand-up
    
    # 2) think() Ersatz / Ergänzung
    def think(self, perception):
        # rufe Stand-up-Logik auf (ohne Parameter)
        self.standing_up()
    
        # rufe Parent-Think (das erzeugt ein action-Objekt)
        action = super(StandingUpAgent, self).think(perception)

        # update last posture for next loop (used by standing_up to detect Sit->Back flips)
        self._last_posture = self.posture
    
        # Falls wir in Stand-Up-Modus sind oder in Post-Hold: setze Steifigkeit an die Motoren
        now = time.time()
        if getattr(self, "standing_up_started", False) or now < self._post_hold_until or now < self._force_stiffness_until:
            try:
                # setze alle Gelenke auf volle Steifigkeit während Aufstehvorgang
                action.stiffness = {j: self._stiffness_value for j in self.joint_names}
            except Exception:
                # falls action kein stiffness-Attribut hat, ignoriere (aber Log wäre hilfreich)
                pass
    
        return action
    
    # 3) vollständige standing_up() ohne Parameter
    def standing_up(self):
        now = time.time()
        posture = self.posture
    
        fallen_postures = ['Back', 'Belly', 'Left', 'Right', 'HeadBack', 'Frog', 'Crouch']
        stand_postures = ['Stand', 'StandInit', 'Sit', 'Knee']
    
        # Reset wenn Stand / Sit erreicht
        if posture in stand_postures:
            if self.standing_up_started:
                print(f"=== Successfully stood up! Now in: {posture} ===")
            self._stand_state = None
            self.standing_up_started = False
            self._post_hold_until = 0.0
            self._force_stiffness_until = 0.0
            return
    
        # Wenn nicht gefallen -> nichts
        if posture not in fallen_postures:
            return
    
        # Falls wir erkannt haben, dass wir kurz vorher SIT hatten und jetzt BACK sind:
        # -> starte aggressive Stabilisierung basierend auf der letzten Keyframe-Sequenz (falls vorhanden)
        last_post = getattr(self, '_last_posture', None)
        state = getattr(self, '_stand_state', None)
        if last_post == 'Sit' and posture == 'Back' and state is not None and not state.get('executing', False):
            # Wenn wir eine letzte Keyframe-Sequenz gespeichert haben, verwende sie als Basis
            last_kf = state.get('last_kf', None)
            if last_kf is not None:
                print(">>> Detected Sit -> Back flip. Launching aggressive stabilization.")
                names, times_list, keys = last_kf
                # größere Vorwärtskorrektur und längeres Halten
                names2, times2, keys2 = _append_stabilize_to_keyframes(names, times_list, keys,
                                                                       hold_time=3, forward_delta=0.15)
                # setze als neue Keyframe-Sequenz
                self.keyframes = (names2, times2, keys2)
                # markiere als executing mit großzügiger Deadline/Puffer
                try:
                    duration = max(max(t) for t in times2)
                except:
                    duration = 12.0
                state['deadline'] = now + duration + 4.0
                state['executing'] = True
                # erzwungene Steifigkeit für die Ausführungs- und Haltezeit
                self._force_stiffness_until = state['deadline'] + 2.0
                self._post_hold_until = state['deadline'] + 2.0
                print(f">>> Executing aggressive stabilization. duration: {duration:.2f}s deadline@{state['deadline']:.2f}")
                return
            # falls keine last_kf vorhanden, fahren wir normal fort und versuchen Sequenzen
            # (kein return hier)
    
        # --- normale initialisierung bei erstem Fallen ---
        if self._stand_state is None:
            # pick primary / alternative by posture
            if posture in ['Back', 'Right', 'HeadBack']:
                primary = rightBackToStand
                alternative = leftBackToStand
            elif posture == 'Left':
                primary = leftBackToStand
                alternative = rightBackToStand
            else:
                primary = rightBellyToStand
                alternative = leftBellyToStand
    
            self._stand_state = {
                'seqs': [(primary, 'primary'), (alternative, 'alternative'), (primary, 'primary')],
                'idx': 0,
                'executing': False,
                'deadline': 0.0,
                'retries': 0,
                'last_try_time': 0.0,
                'last_kf': None
            }
            self.standing_up_started = True
            state = self._stand_state
            print(f"\n=== ROBOT FALLEN! Position: {posture} ===")
            print("=== Starting stand-up sequence... ===\n")
    
        state = self._stand_state
    
        # Prevent tight loop
        if now - state.get('last_try_time', 0.0) < 0.05:
            return
    
        # Wenn bereits eine Sequenz läuft: prüfe Deadline
        if state['executing']:
            if now <= state['deadline']:
                return
            else:
                print(">>> Keyframe attempt timed out or finished without stable stand.")
                state['executing'] = False
                state['retries'] += 1
                state['last_try_time'] = now
                # verlängere post-hold minimal
                self._post_hold_until = now + 1.0
                self._force_stiffness_until = now + 1.0
                if state['retries'] >= self._max_retries:
                    state['idx'] = 0
                    state['retries'] = 0
                    print(">>> Max attempts reached for this fall — pausing briefly before next cycle.")
                    self._post_hold_until = now + 2.0
                    self._force_stiffness_until = now + 2.0
                return
    
        # Wenn alle Versuche schon probiert -> cooldown
        if state['idx'] >= len(state['seqs']):
            print(">>> All fall recovery sequences tried — cooldown before retry.")
            state['idx'] = 0
            state['last_try_time'] = now
            self._post_hold_until = now + 1.0
            self._force_stiffness_until = now + 1.0
            return
    
        # Starte nächste sequence
        seq_fn, tag = state['seqs'][state['idx']]
        state['idx'] += 1
        state['last_try_time'] = now
    
        try:
            names, times_list, keys = seq_fn()
        except Exception as e:
            print(f">>> Error loading keyframes from {seq_fn.__name__}: {e}")
            return
    
        # speichere die letzte keyframe-folge als basis für aggressive stabilization falls nötig
        state['last_kf'] = (names, times_list, keys)
    
        # append stabilization (leise Variante)
        names, times_list, keys = _append_stabilize_to_keyframes(names, times_list, keys, hold_time=2.0, forward_delta=0.10)
    
        # dauer berechnen
        try:
            duration = max(max(t) for t in times_list)
        except:
            duration = 12.0
    
        # setze keyframes und markiere als executing
        self.keyframes = (names, times_list, keys)
        state['deadline'] = now + duration + 3.0
        state['executing'] = True
        self._force_stiffness_until = state['deadline'] + 1.5
        self._post_hold_until = state['deadline'] + 1.5
    
        print(f">>> Executing: {seq_fn.__name__} ({tag})")
        print(f">>> Keyframe sets: {len(names)}  duration: {duration:.2f}s  deadline@{state['deadline']:.2f}")
class TestStandingUpAgent(StandingUpAgent):
    '''this agent turns off all motor to falls down in fixed cycles
    '''
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(TestStandingUpAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.stiffness_on_off_time = 0
        self.stiffness_on_cycle = 10  # in seconds
        self.stiffness_off_cycle = 3  # in seconds

    def think(self, perception):
        action = super(TestStandingUpAgent, self).think(perception)
        time_now = perception.time
        if time_now - self.stiffness_on_off_time < self.stiffness_off_cycle:
            action.stiffness = {j: 0 for j in self.joint_names}  # turn off joints
        else:
            action.stiffness = {j: 1 for j in self.joint_names}  # turn on joints
        if time_now - self.stiffness_on_off_time > self.stiffness_on_cycle + self.stiffness_off_cycle:
            self.stiffness_on_off_time = time_now

        return action


if __name__ == '__main__':
    agent = TestStandingUpAgent()
    agent.run()
