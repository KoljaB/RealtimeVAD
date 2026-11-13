"""Edge cases on real audio. Each test says what it guards and with which tolerance; the
measured values are printed at the end of the run. See conftest.py for where the audio comes from.

"Silero alone" is the same VAD with the quiet clock off, replaying the very same frame
probabilities, so every comparison with it is on identical input.
"""

import numpy as np

import lab

CASES = {}
PAD = ("pad", 0.5, 1.0)  # silence around a clip; reference times shift by OFF
OFF = 0.5


def case(key, parts, **more):
    CASES[key] = {"key": key, "parts": parts, **more}
    return key


def single(prefix, clip, *fx, **more):
    return case(f"{prefix}/{clip['id']}", [(clip["file"], [PAD, *fx])], **more)


def stats(runs, keyed):
    """Per (case key, clean clip): what happened to the reference speech."""
    refs, results = runs
    rows = []
    for key, clip in keyed:
        r, ref = results[key], refs[clip["file"]]
        start = lab.first_event(r, "start")
        closed = all(s["end"] is not None for s in r["segments"] + r["silero_segments"])
        end, end_silero = lab.last_end(r), lab.last_end(r, "silero_segments")
        rows.append({
            "detected": bool(r["segments"]),
            "detected_silero": bool(r["silero_segments"]),
            "covered": lab.covered(r, ref, OFF),
            "covered_silero": lab.covered(r, ref, OFF, "silero_segments"),
            "premature": lab.premature_ends(r, ref, OFF),
            "premature_silero": lab.premature_ends(r, ref, OFF, "silero_segments"),
            "start_delay": None if start is None or ref["on"] is None else start[1] - (ref["on"] + OFF),
            "end_gain": end_silero - end if closed and end is not None and end_silero is not None else None,
            "quiet_ends": sum(s["quiet"] for s in r["segments"]),
        })
    return rows


def total(rows, field):
    return sum(row[field] for row in rows)


def values(rows, field):
    return [row[field] for row in rows if row[field] is not None]


if lab.available():
    SPEECH = lab.clips("speech")  # 8 languages x 10 clips
    S24 = [SPEECH[10 * lang + i] for lang in range(8) for i in (0, 4, 8)]
    S8 = SPEECH[1::10]
    NOISE = {c["id"]: c for c in lab.clips("noise")}
    MUSIC = {c["id"]: c for c in lab.clips("music")}

    CLEAN = [(single("clean", c), c) for c in S24]
    QUIET = {level: [(single(f"quiet{level}", c, ("level", level)), c) for c in S24] for level in (-45, -55)}
    FIXED_FLOOR = [(single("quiet-55-fixed-floor", c, ("level", -55), kw={"floor_margin_db": None}), c) for c in S24]
    PHONE = {level: [(single(f"phone{level}", c, ("level", level), ("phone",)), c) for c in S24] for level in (-30, -45)}
    NOISY = {(kind, snr): [(single(f"{kind}{snr}", c, ("level", -30), ("noise", NOISE[kind + "_1"]["file"], snr)), c) for c in S8]
             for kind in ("babble", "street", "fan") for snr in (20, 10)}
    NOISE_ONLY = [(case(f"noise-only/{c['id']}", [(c["file"], [("cut", 0, 30)])]), c) for c in NOISE.values()]
    MUSIC_ONLY = [(case(f"music-only/{c['id']}", [(c["file"], [("cut", 0, 40)])]), c) for c in MUSIC.values()]
    OVER_MUSIC = [(single("over-music", c, ("level", -30), ("noise", MUSIC["instrumental_piano"]["file"], 10)), c) for c in S8]
    REVERB = {rt60: [(single(f"reverb{rt60}", c, ("reverb", rt60)), c) for c in S8] for rt60 in (0.4, 0.9)}
    NONVERBAL = [(case(f"nonverbal/{c['id']}", [(c["file"], [PAD])]), c) for c in lab.clips("nonverbal")]
    WHISPER = [(single("whisper", c), c) for c in lab.clips("whisper")]
    GATED = [(single("gated", c, ("level", -30), ("white", -65), ("gate", -55)), c) for c in S8]
    CLIPPED = [(single("clipped", c, ("level", -10), ("gain", 25)), c) for c in S8]
    DC = [(single("dc", c, ("dc", 0.02)), c) for c in S8]
    # the default (early_end=False): clean, quiet, phone and noisy clips
    DEFAULT = ([(single("default", c, kw={"early_end": False}), c) for c in S24]
               + [(single("default-55", c, ("level", -55), kw={"early_end": False}), c) for c in S8]
               + [(single("default-phone", c, ("level", -45), ("phone",), kw={"early_end": False}), c) for c in S8]
               + [(single("default-street", c, ("level", -30), ("noise", NOISE["street_1"]["file"], 10), kw={"early_end": False}), c) for c in S8])
    NO_DC = [(single("dc-none", c), c) for c in S8]  # the same clips without the offset
    CLICKS = case("clicks", [(NOISE["fan_1"]["file"], [("cut", 0, 20), ("gain", -200), ("white", -70), ("clicks", 0.7)])])
    BACKCHANNEL = [(case(f"backchannel/{c['id']}", [(c["file"], [])]), c) for c in lab.clips("backchannel")]
    TURN_START = [(case(f"turn-start/{c['id']}", [(c["file"], [])]), c) for c in lab.clips("turn_start")]
    HESITATION = [(case(f"hesitation/{c['id']}", [(c["file"], [("pad", 0, 1.0)])]), c) for c in lab.clips("hesitation")]

    # a soft word ending: the last 0.4 s of the speech fade by 15 dB (set per clip in the test)
    # streams of several utterances
    TALK = [(c["file"], [("level", -30), ("pad", 0.3, 1.5), ("white", -65)]) for c in S24[::4]]
    STREAM = {chunk: case(f"stream/chunk{chunk}", TALK, chunk=chunk) for chunk in (160, 1024, 37)}
    LOUD_THEN_QUIET = [(case(f"loud-then-quiet/{quiet['id']}", [
        (loud["file"], [("level", -22), ("pad", 0.3, 1.0)]), (quiet["file"], [("level", -50), ("pad", 0.3, 1.0)])]), loud, quiet)
        for loud, quiet in zip(S8, S8[1:] + S8[:1])]
    ECHO = [(case(f"echo/{user['id']}", [
        (bot["file"], [("level", -35), ("pad", 0.3, 0.0)]), (NOISE["fan_1"]["file"], [("cut", 0, 2.5), ("gain", -200), ("white", -70)]),
        (user["file"], [("level", -30), ("pad", 0.3, 1.0)])], playback=[(0.0, 0.3 + bot["duration_s"])]), bot, user)
        for bot, user in zip(S8, S8[3:] + S8[:3])]


def test_clean_speech_in_eight_languages(runs, guard):
    """Read sentences, 8 languages, 24 speakers: all found, nothing cut, never later than Silero."""
    rows = stats(runs, CLEAN)
    guard("clips with a segment, of 24", total(rows, "detected"), ">=", 24)
    guard("speech inside segments, worst clip", min(values(rows, "covered")), ">=", 0.97)
    guard("ends decided while the speaker talks", total(rows, "premature"), "<=", 0)
    guard("start reported after the speech began, worst clip", max(values(rows, "start_delay")) * 1000, "<=", 250, "ms")
    guard("last end later than Silero alone, worst clip", max(0.0, -min(values(rows, "end_gain"))) * 1000, "<=", 0, "ms")
    guard.done()


def test_default_ends_where_silero_alone_ends(runs, guard):
    """Without early_end the quiet clock gives the early pause and the speech-off time, never an end:
    every segment starts and ends on the frame Silero alone gives it."""
    _, results = runs
    differ = pauses = leads = segments = 0
    for key, _ in DEFAULT:
        r = results[key]
        mine = [(s["start"], s["end"]) for s in r["segments"]]
        theirs = [(s["start"], s["end"]) for s in r["silero_segments"]]
        differ += mine != theirs
        segments += len(mine)
        pauses += sum(k == "pause" and source == "quiet" for k, _, _, source in r["events"])
        leads += sum(a["speech_end"] is not None and b["speech_end"] is not None and a["speech_end"] < b["speech_end"] - 1e-6
                     for a, b in zip(r["segments"], r["silero_segments"]))
    guard("streams whose segments differ from Silero alone, of 48", differ, "<=", 0)
    guard("pauses from the quiet clock", pauses, ">=", 48)
    guard(f"segments whose speech-off time is before Silero's last voiced frame, of {segments}", leads, ">=", segments // 3)
    guard.done()


def test_quiet_input_does_not_cut_soft_endings(runs, guard):
    """The same sentences with the speech at -45 and -55 dBFS (source-level phone audio).
    A floor fixed at quiet_rms (-54 dBFS) ends segments inside soft word endings here."""
    for level, keyed in QUIET.items():
        rows = stats(runs, keyed)
        guard(f"{level} dBFS: ends decided while the speaker talks", total(rows, "premature"), "<=", total(rows, "premature_silero") + 1)
        behind = [a - b for a, b in zip(values(rows, "covered"), values(rows, "covered_silero"))]
        guard(f"{level} dBFS: speech inside segments vs Silero alone, median clip", float(np.median(behind)), ">=", 0)
        # one noisy recording (16 dB SNR): after an early end its soft next words restart late
        guard(f"{level} dBFS: speech inside segments vs Silero alone, worst clip", min(behind), ">=", -0.1)
        guard(f"{level} dBFS: last end later than Silero alone, worst clip", max(0.0, -min(values(rows, "end_gain"))) * 1000, "<=", 0, "ms")
        guard(f"{level} dBFS: ends by the quiet clock (it still works)", total(rows, "quiet_ends"), ">=", 12)
    fixed = stats(runs, FIXED_FLOOR)  # the test has teeth: the old fixed floor fails it
    guard("-55 dBFS with floor_margin_db=None: premature ends (the old behaviour)", total(fixed, "premature"), ">=", 10)
    guard.done()


def test_phone_line(runs, guard):
    """300-3400 Hz, 8 kHz, mu-law, line noise; speech at -30 and -45 dBFS. The reference is the
    wideband original: fricatives above 3.4 kHz are speech there and gone on the line, so a
    few ends count as premature that the line itself makes (fixed floor: 18 at -45 dBFS)."""
    for level, keyed in PHONE.items():
        rows = stats(runs, keyed)
        guard(f"{level} dBFS: ends decided while the speaker talks", total(rows, "premature"), "<=", {-30: 4, -45: 8}[level])
        # at -45 dBFS Silero stays below 0.5 for two speakers: a known limit of quiet narrowband audio
        guard(f"{level} dBFS: clips with a segment, of 24", total(rows, "detected"), ">=", {-30: 24, -45: 22}[level])
        guard(f"{level} dBFS: last end later than Silero alone, worst clip", max(0.0, -min(values(rows, "end_gain"), default=0)) * 1000, "<=", 0, "ms")
    guard.done()


def test_speech_in_room_noise(runs, guard):
    """Cafe babble, street and fan under the speech at 20 and 10 dB SNR."""
    for (kind, snr), keyed in NOISY.items():
        rows = stats(runs, keyed)
        guard(f"{kind} {snr} dB: clips with a segment, of 8", total(rows, "detected"), ">=", 8)
        guard(f"{kind} {snr} dB: ends decided while the speaker talks", total(rows, "premature"), "<=", total(rows, "premature_silero") + 1)
        guard(f"{kind} {snr} dB: last end later than Silero alone, worst clip", max(0.0, -min(values(rows, "end_gain"), default=0)) * 1000, "<=", 0, "ms")
    guard.done()


def test_noise_alone_is_not_speech(runs, guard):
    """30 s each of fan, street, typing, door, kitchen and household noise, and of babble."""
    _, results = runs
    by_kind = {}
    for key, clip in NOISE_ONLY:
        by_kind.setdefault(clip["id"].rsplit("_", 1)[0], []).append(lab.speech_seconds(results[key]) / results[key]["seconds"])
    for kind, limit in (("typing", 0.0), ("street", 0.0), ("door", 0.0), ("kitchen", 0.0), ("household", 0.0)):
        guard(f"{kind}: share of the time inside a segment, worst file", max(by_kind[kind]), "<=", limit)
    # Silero itself scores one fan recording and one babble recording up to 0.99
    guard("fan: share of the time inside a segment, worst file", max(by_kind["fan"]), "<=", 0.2)
    guard("babble (other people talking): share of the time inside a segment, worst file", max(by_kind["babble"]), "<=", 0.25)
    more = sum(lab.speech_seconds(results[key]) - lab.speech_seconds(results[key], "silero_segments") for key, _ in NOISE_ONLY)
    guard("all noise: seconds inside segments beyond Silero alone", more, "<=", 0.5, "s")
    guard.done()


def test_music(runs, guard):
    """Instrumental music is not speech; speech over music at 10 dB is found."""
    _, results = runs
    for key, clip in MUSIC_ONLY:
        if not clip["singing"]:
            guard(f"{clip['id']}: share of the time inside a segment", lab.speech_seconds(results[key]) / results[key]["seconds"], "<=", 0.05)
    rows = stats(runs, OVER_MUSIC)
    guard("speech over piano at 10 dB: clips with a segment, of 8", total(rows, "detected"), ">=", 8)
    guard("speech over piano: ends decided while the speaker talks", total(rows, "premature"), "<=", total(rows, "premature_silero") + 1)
    guard.done()


def test_reverberant_room(runs, guard):
    """Synthetic rooms with 0.4 s and 0.9 s reverberation time."""
    for rt60, keyed in REVERB.items():
        rows = stats(runs, keyed)
        guard(f"RT60 {rt60} s: clips with a segment, of 8", total(rows, "detected"), ">=", 8)
        guard(f"RT60 {rt60} s: ends decided while the speaker talks", total(rows, "premature"), "<=", total(rows, "premature_silero"))
        guard(f"RT60 {rt60} s: last end later than Silero alone, worst clip", max(0.0, -min(values(rows, "end_gain"), default=0)) * 1000, "<=", 0, "ms")
    guard.done()


def test_coughs_laughter_and_breaths(runs, guard):
    """Human sounds that are not speech. Silero hears most laughter and many coughs as speech;
    the limits pin today's behaviour so a change shows up."""
    _, results = runs
    started = {}
    for key, clip in NONVERBAL:
        started.setdefault(clip["id"].split("_")[0], []).append(bool(results[key]["segments"]))
    for kind, limit in (("sigh", 5), ("sneeze", 8), ("throat", 9), ("cough", 7), ("laughter", 10)):
        guard(f"{kind}: clips that start a segment, of 10", sum(started[kind]), "<=", limit)
    guard.done()


def test_whispering(runs, guard):
    """Whispered Hindi sentences. A known limit: Silero scores most whisper below 0.5, so
    most of it is not found. Pins that the little that is found stays found."""
    rows = stats(runs, WHISPER)
    guard("whispered clips with a segment, of 10", total(rows, "detected"), ">=", 2)
    guard("ends decided while the speaker talks", total(rows, "premature"), "<=", total(rows, "premature_silero") + 1)
    guard.done()


def test_muting_noise_suppression(runs, guard):
    """A noise gate turns everything below -55 dBFS into digital zero."""
    rows = stats(runs, GATED)
    guard("clips with a segment, of 8", total(rows, "detected"), ">=", 8)
    guard("ends decided while the speaker talks", total(rows, "premature"), "<=", total(rows, "premature_silero"))
    guard("last end later than Silero alone, worst clip", max(0.0, -min(values(rows, "end_gain"))) * 1000, "<=", 0, "ms")
    guard.done()


def test_clipping_and_dc_offset(runs, guard):
    """Speech 15 dB into clipping, and a DC offset of -34 dBFS under the speech (levels are
    measured without it, so the quiet clock ends the segments as it does without the offset)."""
    for name, keyed in (("clipped", CLIPPED), ("DC offset", DC)):
        rows = stats(runs, keyed)
        guard(f"{name}: clips with a segment, of 8", total(rows, "detected"), ">=", 8)
        guard(f"{name}: speech inside segments vs Silero alone, worst clip",
              min(a - b for a, b in zip(values(rows, "covered"), values(rows, "covered_silero"))), ">=", -0.02)
        guard(f"{name}: ends decided while the speaker talks", total(rows, "premature"), "<=", total(rows, "premature_silero"))
        guard(f"{name}: last end later than Silero alone, worst clip", max(0.0, -min(values(rows, "end_gain"), default=0)) * 1000, "<=", 0, "ms")
    _, results = runs
    plain = stats(runs, NO_DC)
    guard("DC offset: ends by the quiet clock, of those without the offset", total(rows, "quiet_ends"), ">=", total(plain, "quiet_ends"))
    guard("DC offset: ends by the quiet clock", total(rows, "quiet_ends"), ">=", 8)
    moved = max(abs(lab.last_end(results[a]) - lab.last_end(results[b])) for (a, _), (b, _) in zip(DC, NO_DC))
    # Silero itself hears the offset: its probabilities, and so an end, can move by a frame
    guard("DC offset: last end moved against the same clip without it, worst clip", moved * 1000, "<=", 33, "ms")
    guard.done()


def test_clicks_do_not_start_a_segment(runs, guard):
    """A 2 ms full-scale click every 0.7 s over 20 s of faint noise."""
    _, results = runs
    guard("segments", len(results[CLICKS]["segments"]), "<=", 0)
    guard.done()


def test_backchannels_are_heard(runs, guard):
    """200 real "mm-hmm", "yeah", "okay" from meetings (AMI, close microphone), 0.15-0.8 s long.
    Only the VAD question: is such a short utterance reported as speech, and how soon."""
    _, results = runs
    heard, delays, alone = [], [], []
    for key, clip in BACKCHANNEL:
        a, b = clip["speech_start_s"], clip["speech_end_s"]
        starts = [emitted for kind, _, emitted, _ in results[key]["events"] if kind == "start" and a - 0.2 <= emitted <= b + 0.3]
        # crosstalk from the others can have opened a segment already; then the word is inside it
        running = any(s["start"] < a - 0.2 and (s["end"] is None or s["end"] > (a + b) / 2) for s in results[key]["segments"])
        heard.append(bool(starts) or running)
        if starts:
            delays.append(starts[0] - a)
        if not clip["others_talking"]:
            alone.append(bool(starts) or running)
    # without the soft start (soft_start_s=None) 150 and 29 are heard; Silero stays below 0.5 for 15
    guard("backchannels reported as speech, of 200", sum(heard), ">=", 170)
    guard("... of the 45 said while nobody else talks", sum(alone), ">=", 36)
    guard("start reported after the word began, median", float(np.median(delays)) * 1000, "<=", 250, "ms")
    guard.done()


def test_turn_starts_are_heard(runs, guard):
    """60 real turn beginnings from the same meetings."""
    _, results = runs
    delays = []
    for key, clip in TURN_START:
        starts = [emitted for kind, _, emitted, _ in results[key]["events"] if kind == "start"
                  and clip["speech_start_s"] - 0.2 <= emitted <= clip["speech_end_s"]]
        if starts:
            delays.append(starts[0] - clip["speech_start_s"])
    guard("turn starts with a start, of 60", len(delays), ">=", 57)
    guard("start reported after the turn began, median", float(np.median(delays)) * 1000, "<=", 250, "ms")
    guard.done()


def test_hesitations_keep_their_words(runs, guard):
    """30 real turns with a 0.5-1.5 s pause in the middle: the pause may end a segment (that is
    what a VAD does), but the words after it must be inside a segment again."""
    _, results = runs
    kept = []
    for key, clip in HESITATION:
        r = results[key]
        inside = spoken = 0.0
        for _, a, b in clip["words"]:
            spoken += b - a
            inside += sum(max(0.0, min(b, s["end"] if s["end"] is not None else 1e9) - max(a, s["from"])) for s in r["segments"])
        kept.append(inside / spoken)
    guard("word time inside segments, median turn", float(np.median(kept)), ">=", 0.95)
    guard("word time inside segments, worst turn", min(kept), ">=", 0.6)
    guard.done()


def test_long_stream_and_chunk_sizes(runs, guard):
    """Six utterances in six languages in one stream; fed in 10 ms, 64 ms and 37-sample chunks."""
    refs, results = runs
    r = results[STREAM[160]]
    found = sum(lab.covered(r, refs[file], offset + 0.3) > 0.9 for (file, _), offset in zip(TALK, r["offsets"]))
    guard("utterances found, of 6", found, ">=", 6)
    gains = []
    for (file, _), offset in zip(TALK, r["offsets"]):  # each utterance's end against Silero alone, over -65 dBFS noise
        off = refs[file]["off"] + offset + 0.3
        mine = [s["end"] for s in r["segments"] if s["end"] is not None and off <= s["end"] <= off + 1.0]
        theirs = [s["end"] for s in r["silero_segments"] if s["end"] is not None and off <= s["end"] <= off + 1.0]
        if mine and theirs:
            gains.append(theirs[0] - mine[0])
    guard("utterance ends compared with Silero alone, of 6", len(gains), ">=", 5)
    guard("end earlier than Silero alone, median", float(np.median(gains)) * 1000, ">=", 30, "ms")
    guard("end later than Silero alone, worst", max(0.0, -min(gains)) * 1000, "<=", 0, "ms")
    for chunk in (1024, 37):
        other = results[STREAM[chunk]]
        same = len(other["segments"]) == len(r["segments"]) and all(
            abs(a["start"] - b["start"]) < 1e-6 and abs(a["end"] - b["end"]) < 1e-6 for a, b in zip(r["segments"], other["segments"]))
        guard(f"chunks of {chunk} samples give the same segments", int(same), ">=", 1)
    guard.done()


def test_quiet_speaker_after_a_loud_one(runs, guard):
    """A speaker at -22 dBFS, then another at -50 dBFS: the floor has to come down in time."""
    refs, results = runs
    premature = premature_silero = found = 0
    for key, _, quiet in LOUD_THEN_QUIET:
        r, ref, offset = results[key], refs[quiet["file"]], results[key]["offsets"][1] + 0.3
        premature += lab.premature_ends(r, ref, offset)
        premature_silero += lab.premature_ends(r, ref, offset, "silero_segments")
        found += lab.covered(r, ref, offset) > 0.9
    guard("quiet second speakers found, of 8", found, ">=", 7)
    guard("ends decided while the quiet speaker talks", premature, "<=", premature_silero + 1)
    guard.done()


def test_after_the_assistants_own_voice(runs, guard):
    """Playback echo (marked with playback=True) at -35 dBFS, 2.5 s of quiet, then the user."""
    refs, results = runs
    phantom = found = 0
    for key, bot, user in ECHO:
        r = results[key]
        quiet_from, user_at = r["offsets"][1], r["offsets"][2]
        phantom += sum(quiet_from + 0.3 < s["start"] < user_at for s in r["segments"])
        found += lab.covered(r, refs[user["file"]], user_at + 0.3) > 0.9
    guard("segments that start in the quiet after playback", phantom, "<=", 0)
    guard("user utterances after playback found, of 8", found, ">=", 8)
    guard.done()
