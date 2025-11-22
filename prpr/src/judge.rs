//! Judgement system

use crate::{
    config::Config,
    core::{BadNote, Chart, NoteKind, Point, Resource, Vector, NOTE_WIDTH_RATIO_BASE},
    ext::{get_viewport, NotNanExt},
};
use macroquad::prelude::{
    utils::{register_input_subscriber, repeat_all_miniquad_input},
    *,
};
use miniquad::{EventHandler, MouseButton};
use once_cell::sync::Lazy;
use sasa::{PlaySfxParams, Sfx};
use serde::Serialize;
use std::{cell::RefCell, collections::HashMap, num::FpCategory};
use tracing::debug;

pub const FLICK_SPEED_THRESHOLD: f32 = 0.8;
pub const LIMIT_PERFECT: f32 = 0.04;
pub const LIMIT_GOOD: f32 = 0.075;
pub const LIMIT_BAD: f32 = 0.22;
pub const UP_TOLERANCE: f32 = 0.05;
pub const DIST_FACTOR: f32 = 0.2;

#[derive(Debug, Clone)]
pub enum HitSound {
    None,
    Click,
    Flick,
    Drag,
    Custom(String),
}

impl HitSound {
    pub fn play(&self, res: &mut Resource) {
        match self {
            HitSound::None => {}
            HitSound::Click => play_sfx(&mut res.sfx_click, &res.config),
            HitSound::Flick => play_sfx(&mut res.sfx_flick, &res.config),
            HitSound::Drag => play_sfx(&mut res.sfx_drag, &res.config),
            HitSound::Custom(s) => {
                if let Some(sfx) = res.extra_sfxs.get_mut(s) {
                    play_sfx(sfx, &res.config);
                }
            }
        }
    }

    pub fn default_from_kind(kind: &NoteKind) -> Self {
        match kind {
            NoteKind::Click => HitSound::Click,
            NoteKind::Flick => HitSound::Flick,
            NoteKind::Drag => HitSound::Drag,
            NoteKind::Hold { .. } => HitSound::Click,
        }
    }
}

pub fn play_sfx(sfx: &mut Sfx, config: &Config) {
    if config.volume_sfx <= 1e-2 {
        return;
    }
    let _ = sfx.play(PlaySfxParams {
        amplifier: config.volume_sfx,
    });
}

#[cfg(all(not(target_os = "windows"), not(target_os = "ios")))]
fn get_uptime() -> f64 {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    let ret = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    assert!(ret == 0);
    time.tv_sec as f64 + time.tv_nsec as f64 * 1e-9
}

#[cfg(target_os = "ios")]
fn get_uptime() -> f64 {
    use crate::objc::*;
    unsafe {
        let process_info: ObjcId = msg_send![class!(NSProcessInfo), processInfo];
        msg_send![process_info, systemUptime]
    }
}

pub struct FlickTracker {
    threshold: f32,
    last_point: Point,
    last_delta: Option<Vector>,
    last_time: f32,
    flicked: bool,
    stopped: bool,
}

impl FlickTracker {
    pub fn new(dpi: u32, time: f32, point: Point) -> Self {
        Self {
            threshold: FLICK_SPEED_THRESHOLD * dpi as f32 / 386.,
            last_point: point,
            last_delta: None,
            last_time: time,
            flicked: false,
            stopped: true,
        }
    }

    pub fn push(&mut self, time: f32, position: Point) {
        let delta = position - self.last_point;
        self.last_point = position;
        if let Some(last_delta) = &self.last_delta {
            let dt = time - self.last_time;
            let speed = delta.dot(last_delta) / dt;
            if speed < self.threshold {
                self.stopped = true;
            }
            if self.stopped && !self.flicked {
                self.flicked = delta.magnitude() / dt >= self.threshold * 2.;
            }
        }
        self.last_delta = Some(delta.normalize());
        self.last_time = time;
    }
}

#[derive(Debug)]
pub enum JudgeStatus {
    NotJudged,
    PreJudge,
    Judged,
    Hold(bool, f32, f32, bool, f32),
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, Serialize)]
pub enum Judgement {
    Perfect,
    Good,
    Bad,
    Miss,
}

#[cfg(not(feature = "closed"))]
#[derive(Default)]
pub(crate) struct JudgeInner {
    diffs: Vec<f32>,

    combo: u32,
    max_combo: u32,
    counts: [u32; 4],
    num_of_notes: u32,
}

#[cfg(not(feature = "closed"))]
impl JudgeInner {
    pub fn new(num_of_notes: u32) -> Self {
        Self {
            diffs: Vec::new(),

            combo: 0,
            max_combo: 0,
            counts: [0; 4],
            num_of_notes,
        }
    }

    pub fn commit(&mut self, what: Judgement, diff: f32) {
        use Judgement::*;
        if matches!(what, Judgement::Good) {
            self.diffs.push(diff);
        }
        self.counts[what as usize] += 1;
        match what {
            Perfect | Good => {
                self.combo += 1;
                if self.combo > self.max_combo {
                    self.max_combo = self.combo;
                }
            }
            _ => {
                self.combo = 0;
            }
        }
    }

    pub fn reset(&mut self) {
        self.combo = 0;
        self.max_combo = 0;
        self.counts = [0; 4];
        self.diffs.clear();
    }

    pub fn accuracy(&self) -> f64 {
        (self.counts[0] as f64 + self.counts[1] as f64 * 0.65) / self.num_of_notes as f64
    }

    pub fn real_time_accuracy(&self) -> f64 {
        let cnt = self.counts.iter().sum::<u32>();
        if cnt == 0 {
            return 1.;
        }
        (self.counts[0] as f64 + self.counts[1] as f64 * 0.65) / cnt as f64
    }

    pub fn score(&self) -> u32 {
        const TOTAL: u32 = 1000000;
        if self.counts[0] == self.num_of_notes {
            TOTAL
        } else {
            let score = (0.9 * self.accuracy() + self.max_combo as f64 / self.num_of_notes as f64 * 0.1) * TOTAL as f64;
            score.round() as u32
        }
    }

    pub fn result(&self) -> PlayResult {
        let early = self.diffs.iter().filter(|it| **it < 0.).count() as u32;
        PlayResult {
            score: self.score(),
            accuracy: self.accuracy(),
            max_combo: self.max_combo,
            num_of_notes: self.num_of_notes,
            counts: self.counts,
            early,
            late: self.diffs.len() as u32 - early,
            std: 0.,
        }
    }

    pub fn combo(&self) -> u32 {
        self.combo
    }

    pub fn counts(&self) -> [u32; 4] {
        self.counts
    }
}

#[cfg(feature = "closed")]
pub mod inner;
#[cfg(feature = "closed")]
use inner::*;

#[repr(C)]
pub struct Judge {
    pub notes: Vec<(Vec<u32>, usize)>,
    pub trackers: HashMap<u64, FlickTracker>,
    pub last_time: f32,

    key_down_count: u32,

    pub(crate) inner: JudgeInner,
    pub judgements: RefCell<Vec<(f32, u32, u32, Result<Judgement, bool>)>>,
}

static SUBSCRIBER_ID: Lazy<usize> = Lazy::new(register_input_subscriber);
thread_local! {
    static TOUCHES: RefCell<(Vec<Touch>, i32, u32)> = RefCell::default();
}

impl Judge {
    pub fn new(chart: &Chart) -> Self {
        let notes = chart
            .lines
            .iter()
            .map(|line| {
                let mut idx: Vec<u32> = (0..(line.notes.len() as u32)).filter(|it| !line.notes[*it as usize].fake).collect();
                idx.sort_by_key(|id| line.notes[*id as usize].time.not_nan());
                (idx, 0)
            })
            .collect();
        Self {
            notes,
            trackers: HashMap::new(),
            last_time: 0.,

            key_down_count: 0,

            inner: JudgeInner::new(chart.lines.iter().map(|it| it.notes.iter().filter(|it| !it.fake).count() as u32).sum()),
            judgements: RefCell::new(Vec::new()),
        }
    }

    pub fn reset(&mut self) {
        self.notes.iter_mut().for_each(|it| it.1 = 0);
        self.trackers.clear();
        self.inner.reset();
        self.judgements.borrow_mut().clear();
    }

    pub fn commit(&mut self, t: f32, what: Judgement, line_id: u32, note_id: u32, diff: f32) {
        self.judgements.borrow_mut().push((t, line_id, note_id, Ok(what)));
        self.inner.commit(what, diff);
    }

    #[inline]
    pub fn accuracy(&self) -> f64 {
        self.inner.accuracy()
    }

    #[inline]
    pub fn real_time_accuracy(&self) -> f64 {
        self.inner.real_time_accuracy()
    }

    #[inline]
    pub fn score(&self) -> u32 {
        self.inner.score()
    }

    pub(crate) fn on_new_frame() {
        let mut handler = Handler(Vec::new(), 0, 0);
        repeat_all_miniquad_input(&mut handler, *SUBSCRIBER_ID);
        handler.finalize();
        TOUCHES.with(|it| {
            *it.borrow_mut() = (handler.0, handler.1, handler.2);
        });
    }

    fn touch_transform(flip_x: bool) -> impl Fn(&mut Touch) {
        let vp = get_viewport();
        move |touch| {
            let p = touch.position;
            touch.position = vec2(
                (p.x - vp.0 as f32) / vp.2 as f32 * 2. - 1.,
                ((p.y - (screen_height() - (vp.1 + vp.3) as f32)) / vp.3 as f32 * 2. - 1.) / (vp.2 as f32 / vp.3 as f32),
            );
            if flip_x {
                touch.position.x *= -1.;
            }
        }
    }

    pub fn get_touches() -> Vec<Touch> {
        TOUCHES.with(|it| {
            let guard = it.borrow();
            let tr = Self::touch_transform(false);
            guard
                .0
                .iter()
                .cloned()
                .map(|mut it| {
                    tr(&mut it);
                    it
                })
                .collect()
        })
    }

    pub fn update(&mut self, res: &mut Resource, chart: &mut Chart, bad_notes: &mut Vec<BadNote>) {
        if res.config.autoplay() {
            self.auto_play_update(res, chart);
            return;
        }
        let base_x_diff_max: f32 = 0.21 / (16. / 9.) * 2. * 1.15;
        let spd = res.config.speed;
        let x_hit_limit = base_x_diff_max;

        #[cfg(not(target_os = "windows"))]
        let uptime = get_uptime();

        let t = res.time;
        let mut touches: HashMap<u64, Touch> = {
            let mut touches = touches();
            let btn = MouseButton::Left;
            let id = button_to_id(btn);
            if is_mouse_button_pressed(btn) {
                let p = mouse_position();
                touches.push(Touch {
                    id,
                    phase: TouchPhase::Started,
                    position: vec2(p.0, p.1),
                    time: f64::NEG_INFINITY,
                });
            } else if is_mouse_button_down(btn) {
                let p = mouse_position();
                touches.push(Touch {
                    id,
                    phase: TouchPhase::Moved,
                    position: vec2(p.0, p.1),
                    time: f64::NEG_INFINITY,
                });
            } else if is_mouse_button_released(btn) {
                let p = mouse_position();
                touches.push(Touch {
                    id,
                    phase: TouchPhase::Ended,
                    position: vec2(p.0, p.1),
                    time: f64::NEG_INFINITY,
                });
            }
            let tr = Self::touch_transform(res.config.flip_x());
            touches
                .into_iter()
                .map(|mut it| {
                    tr(&mut it);
                    (it.id, it)
                })
                .collect()
        };
        let (events, keys_down) = TOUCHES.with(|it| {
            let guard = it.borrow();
            (guard.0.clone(), guard.2)
