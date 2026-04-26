# We Built an RL Env Where the Model Has to Deal With Executive Chaos

okay so the obvious idea for "personalized tasks" is email triage right

classify spam, mark urgent, maybe write a reply. everyone does it. it's the first thing that comes to mind.

we didn't want to do that.

we wanted to build something where the model actually has to *think* — not just pattern match on obvious signals.

so we sat down and asked: what does an executive assistant actually deal with every morning?

turns out it's not emails. it's conflicts.

your 10:30 board meeting just got moved and now it's sitting on top of the client demo you locked in last week. a contract expires at 5pm and you've been in back-to-back calls since 8am. someone invited you to a team dinner that starts right when your investor call runs until. a client is about to walk and their account manager is on leave with zero backup.

none of these have a clean answer. you can't just slap a label on them. you have to actually figure out what's going on, decide what to do, and then say something useful to real people.

that's what we built.

---

## how it works

every time you hit reset, the model gets a fresh inbox. 5 conflict situations. different people, different companies, different stakes, different urgency. fully randomized every run so no two episodes are the same.

for each conflict the model has to answer three things:

what kind of conflict is this — scheduling clash, deadline, delegation issue, or social obligation

what's the actual move — reschedule, decline, delegate, accept, or escalate

and then... what do you say

that last one is where it gets interesting. the model has to write something like:

```
{
  "item_id": "c2",
  "conflict_type": "deadline",
  "resolution": "escalate",
  "message": "hey flagging this as urgent — the vertex ai contract expires at 10:30 and i'm stuck in calls all morning. can someone get sign-off authority or loop in the CEO directly?"
}
```

not a template. something that actually fits the specific situation it was given.

we split this into three levels — easy just asks for the conflict type, medium adds the resolution, hard makes you write the message too. hard is where it stops being a toy.

---

## scoring

conflict type — exact match. right is 0.99, wrong is 0.01. pretty unforgiving.

resolution — a bit more flexible. some situations genuinely have more than one valid answer. escalating when the "expected" answer is delegating still gets full credit because honestly both make sense in context.

message — we check for the signals you'd expect. reschedule message should mention availability. decline should acknowledge the invite. escalation should feel urgent. not grading grammar, just whether the message actually fits what you said you'd do.

leave something unresolved — you get penalized.

---

## training

we trained qwen2.5-1.5b-instruct using grpo via trl, lora adapters on a 4-bit base. nothing fancy.

the reward wasn't simulated though — every output got sent to the actual live environment, scored in real time, and fed back into training. we wanted the model learning against real environment behavior not a proxy.

100 prompts, 2 epochs, ~100 steps. reward averaged over 3 env calls per completion to smooth out noise.

baseline scores before any training:

| task | score |
|------|-------|
| easy | 0.99 |
| medium | 0.81 |
| hard | 0.39 |

easy is basically solved out of the box. medium is decent. hard is where reality hits — 0.39 means the model sometimes gets it right but not consistently.

![Training Results](https://raw.githubusercontent.com/notAathi/OpenEnvHack/main/training_summary.png)

---

## what actually made this hard

honestly the hardest part wasn't the model. it was the environment itself.

because the reward comes from a live api, every training step is slightly different. different inbox, slight scoring variation, network latency. the same response can get different rewards across calls just because the conflict items are randomized each time.

that noise compounds fast. in under 200 steps the model doesn't get enough clean signal to settle into a stable policy. it starts chasing noise instead of learning the actual task.

the environment works. the reward function is meaningful. what's missing is more compute, longer runs, and better noise handling. all solvable — just not in one hackathon session.

---

## try it

environment is live. hit reset and you get a different messy inbox every time.

live env → https://huggingface.co/spaces/notAathi/OpenEvnHack

github → https://github.com/notAathi/OpenEnvHack

training notebook → https://www.kaggle.com/code/notaathi/openenvaathi

trained model → https://huggingface.co/notAathi/conflict-resolution-grpo

---

we didn't want to build another benchmark. we wanted to build something that feels like actual work — the messy overlapping time-sensitive stuff real assistants deal with every day.

if llms are going to be genuinely useful they shouldn't just sort emails. they should be able to walk into chaos and figure out what to do next.

*— notAathi*
