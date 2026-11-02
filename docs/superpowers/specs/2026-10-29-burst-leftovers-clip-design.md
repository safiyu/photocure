# Burst leftovers become a clip

Date: 2026-10-29
Status: roadmap draft, parked. Not yet reviewed by a builder, not approved.

## Intent

When a burst has spare frames, photocure's answer today is to pick the best
shot and offer to delete the rest. The rest are not worthless: together they
hold the half second of motion around the keeper. This feature turns the whole
burst into a short stabilised loop, saved next to the best shot and shown in
the Library as its own item. The frames you were about to throw away become
the memory of the moment.

The owner chose this from a list of ideas on 2026-10-29 after seeing a demo
built from two real bursts in their library. Of three styles shown (raw
flipbook, boomerang, stabilised loop) they chose the stabilised loop, and
chose that the clip should be a first class Library item.

## Goals

- From any burst group, produce one short looping clip: every frame aligned
  to the best shot, lightly cropped to hide warped edges, crossfaded between
  frames, holding longest on the best shot.
- The clip is saved next to the best shot and never touches the originals,
  following the convention of `IMG_xxxx_fixed.jpg`.
- The clip appears in the Library next to the best shot with a play badge,
  inherits the best shot's date, place, people, category and event, can be
  starred and favourited, and is eligible for Highlights and event stories.
- The clip survives the spare frames being trashed or permanently deleted.
- Everything stays local and reversible, like the rest of photocure.

## Non-goals

- Replacing the best shot. The still keeper stays the primary item.
- Video editing of any kind: no trimming, speed control, music or titles.
- Working on arbitrary selections of photos. Input is always a burst group.
- Changing how bursts are detected or how the best shot is chosen.

## Anti-goals

- A clip that looks like a slideshow of rejects. If alignment fails for most
  frames, the feature should decline rather than produce a jumpy result.
- Silent background work that fills the library with clips nobody asked for.
  The owner did not choose automatic creation; the clip is made on request.
- Cloud or network dependence. No uploads, no remote encoders.

## What the demo established

A throwaway spike (`/tmp/pc_spike/leftovers_clip.py`, not in the repo) built
clips from bursts 11 and 20 of the owner's current library using Pillow and
OpenCV only, both already installed in the project venv. ORB feature matching
with a partial affine fit aligned frames well enough even though those frames
were 10 to 20 seconds apart. A true camera burst will align far better.

Sizes at 600 by 400 pixels, useful for the format decision below:

| Style | Animated WebP | GIF |
|---|---|---|
| Flipbook, 5 frames | 271 KB | 864 KB |
| Stabilised loop, 25 frames | 790 KB | 3.6 MB |

ffmpeg is not installed on the owner's machine and must not be assumed.

## Decisions taken by the owner

- Style: stabilised loop (option C in the demo).
- The clip is a Library item in its own right (option F in the demo).

## Calls made in this document

These are the author's defaults. Change them freely before planning.

- **Trigger.** A **Make clip** button on the burst card in the Bursts page,
  next to **Create fixed photo**. It is enabled when the burst has at least
  three frames on disk and alignment succeeds for at least two thirds of them.
- **File name.** `<best>_clip.<ext>` next to the best shot, using the same
  unique naming helper as the fixed photo.
- **Playback timing.** Best shot held about 0.9 s, other frames about 0.4 s,
  four crossfade steps of 50 ms between frames, loop forever.
- **Size.** Long edge 1080 px by default. The source frames are untouched.
- **Alignment.** ORB keypoints plus partial affine with RANSAC, as in the
  spike. Fall back to the unaligned frame when matching fails, and refuse the
  whole clip when fewer than two thirds of frames align.
- **Library row.** The clip gets its own image row with a new `derived_from`
  column pointing at the best shot and a `kind` of `clip`. It copies category,
  taken_at, GPS, event, people and keywords from the best shot at creation
  time. Deleting the best shot does not delete the clip.
- **Apply and undo.** The clip is filed into the library folder in the same
  plan step as its best shot. Undo removes it along with the rest of the plan.
  Trashing the spare frames afterwards has no effect on the clip.
- **Thumbnails.** The clip's thumbnail is the best shot with a play badge
  overlay, so the grid stays fast and no decoder is needed for the list view.
- **Scanner.** The chosen clip extension is added to the scanner's accepted
  list only for files with the `_clip` suffix, so unrelated GIFs and videos in
  source folders are still ignored.

## Open question that blocks planning

**File format.** This is the owner's call and changes what gets built.

| Option | For | Against |
|---|---|---|
| Animated WebP | Small, good colour, Pillow writes it today, every browser plays it | Finder Quick Look and some photo apps show only the first frame |
| Animated GIF | Opens absolutely everywhere | 256 colours, four to five times larger |
| MP4 (H.264) | Smallest, universal, plays in Photos, Finder, phones | Needs an encoder; OpenCV's bundled writer is unreliable across platforms and ffmpeg is not installed |
| Apple Live Photo pair | Native on iPhone and Photos | Only meaningful on Apple devices, needs a paired JPEG plus MOV with matching identifiers |

Recommendation: ship animated WebP first because it needs nothing new, and
add MP4 as an optional output when ffmpeg or imageio-ffmpeg is present.

## Constraints

- Local only, Python 3.11, Pillow and OpenCV already present.
- Must work on CPU; the GPU is optional.
- Never modify or delete any source frame.
- Library rows for finished photos are never re-analysed, so the clip must be
  fully described at creation time.

## Rough roadmap

1. **Core clip builder.** Pure function from frame paths and best index to an
   encoded clip, with the alignment and refusal rules above. Unit tests on
   synthetic shifted frames.
2. **Bursts page button and endpoint.** Make clip, show progress, show the
   result inline on the burst card.
3. **Library integration.** New row kind, inherited metadata, play badge
   thumbnail, viewer playback, apply and undo.
4. **Highlights and stories.** Allow the clip to stand in for its best shot
   in event stories when the owner opts in.
5. **Optional MP4 output** when an encoder is available.

## What is left to the builder

- Exact UI copy and placement on the burst card.
- Crop margin and crossfade counts, tuned on real bursts.
- How the viewer plays the clip (native `<img>` for WebP and GIF, `<video>`
  for MP4).
- Migration for the new `derived_from` and `kind` columns.
