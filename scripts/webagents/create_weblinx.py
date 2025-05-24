
import datasets
import minari
from minari.data_collector import EpisodeBuffer
import gymnasium as gym


def main():
    dataset = datasets.load_dataset("McGill-NLP/WebLINX", split="train")
    current_demo = None
    episodes = []
    demos = []

    current_action = None
    for demo, action, action_history, utterances, candidates, clean_html, viewport in zip(
        dataset["demo"], dataset["action"], dataset["action_history"], dataset["utterances"], dataset["candidates"], dataset["clean_html"], dataset["viewport"]
    ):
        if current_demo != demo:
            current_demo = demo
            demos.append(current_demo)

            if len(episodes):
                episodes[-1] = episodes[-1].add_step_data(dict(
                    observation={
                        "action_history": "",
                        "candidates": "",
                        "utterances": "",
                        "viewport": "",
                        "clean_html": ""
                    },
                    action=current_action,
                    reward=0,
                    terminated=True,
                    truncated=False,
                    info=None
                ))
            
            new_episode = EpisodeBuffer(
                observations={
                    "action_history": action_history or "",
                    "candidates": candidates or "",
                    "utterances": utterances or "",
                    "viewport": viewport or "",
                    "clean_html": clean_html or ""
                },
            )
            episodes.append(new_episode)
            current_action = action
        else:
            episodes[-1] = episodes[-1].add_step_data(dict(
                observation={
                    "action_history": action_history or "",
                    "candidates": candidates or "",
                    "utterances": utterances or "",
                    "viewport": viewport or "",
                    "clean_html": clean_html or ""
                },
                action=current_action,
                reward=0,
                terminated=False,
                truncated=False,
                info=None
            ))

            current_action = action
        
    minari.create_dataset_from_buffers(
        dataset_id="webagents/weblinx-v0",
        buffer=episodes,
        author={"Xing Han Lù", "Zdeněk Kasner", "Siva Reddy"},
        author_email="xing.han.lu@mail.mcgill.ca",
        code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts",
        action_space=gym.spaces.Text(max_length=1024),
        observation_space=gym.spaces.Dict({
            "action_history": gym.spaces.Text(max_length=1024 * 16),
            "clean_html": gym.spaces.Text(max_length=1024 * 16),
            "candidates": gym.spaces.Text(max_length=1024 * 16),
            "utterances": gym.spaces.Text(max_length=1024 * 8),
            "viewport": gym.spaces.Text(max_length=1024)
        })
    )


if __name__ == "__main__":
    main()


