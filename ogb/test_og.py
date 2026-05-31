import ogbench
import cv2

# Make an environment and datasets (they will be automatically downloaded).
dataset_name = 'pointmaze-medium-navigate-v0'  # Choose from 'cube-single-object-v0', 'cube-multi-object-v0', 'drawer-single-object-v0', 'drawer-multi-object-v0'.
env = ogbench.make_env_and_datasets(dataset_name, env_only=True)

# Train your offline goal-conditioned RL agent on the dataset.
# ...
render = []
# Evaluate the agent.
for task_id in [1]:
    # Reset the environment and set the evaluation task.
    ob, info = env.reset(
        options=dict(
            task_id=task_id,  # Set the evaluation task. Each environment provides five
                              # evaluation goals, and `task_id` must be in [1, 5].
            render_goal=True,  # Set to `True` to get a rendered goal image (optional).
        )
    )

    goal = info['goal']  # Get the goal observation to pass to the agent.
    goal_rendered = info['goal_rendered']  # Get the rendered goal image (optional).

    done = False
    while not done:
        action = env.action_space.sample()  # Replace this with your agent's action.
        ob, reward, terminated, truncated, info = env.step(action)  # Gymnasium-style step.
        # If the agent reaches the goal, `terminated` will be `True`. If the episode length
        # exceeds the maximum length without reaching the goal, `truncated` will be `True`.
        # `reward` is 1 if the agent reaches the goal and 0 otherwise.
        done = terminated or truncated
        frame = env.render()  # Render the current frame (optional).
        render.append(frame)

    success = info['success']  # Whether the agent reached the goal (0 or 1).
                               # `terminated` also indicates this.
    print(f'Task {task_id} success: {success}')
env.close()

cv2.namedWindow(dataset_name, cv2.WINDOW_NORMAL)
for i in range(len(render)):
    cv2.imshow(dataset_name, render[i])
    cv2.waitKey(100)
cv2.destroyAllWindows()