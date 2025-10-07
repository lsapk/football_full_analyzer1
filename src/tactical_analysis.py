import os
import openai

def format_data_for_llm(team_stats_df, player_stats_df, events_df):
    """
    Formats the analysis data into a structured prompt for a Large Language Model.
    """
    prompt = "You are an expert football tactical analyst. Based on the following match data, provide a tactical analysis report. \n"
    prompt += "Focus on each team's strategy, strengths, weaknesses, and suggest one key improvement for each.\n\n"

    # --- Team Stats ---
    prompt += "# Team Statistics\n"
    prompt += "Here are the overall statistics for each team:\n"
    prompt += team_stats_df.to_markdown(index=True)
    prompt += "\n\n"

    # --- Player Stats ---
    prompt += "# Key Player Statistics\n"
    prompt += "Here are the top 3 players from each team based on distance covered:\n"
    for team_id in player_stats_df['team_id'].unique():
        team_name = team_stats_df.loc[team_id]['team_name']
        top_players = player_stats_df[player_stats_df['team_id'] == team_id].nlargest(3, 'distance_m')
        prompt += f"## {team_name}:\n"
        prompt += top_players[['player_id', 'distance_m', 'max_speed_kmh', 'touches']].to_markdown(index=False)
        prompt += "\n"
    prompt += "\n"

    # --- Event Summary ---
    prompt += "# Event Summary\n"
    if not events_df.empty:
        event_summary = events_df.groupby('team_id')['type'].value_counts().unstack(fill_value=0)
        prompt += "Here is a summary of key events (passes, dribbles) initiated by each team:\n"
        prompt += event_summary.to_markdown()
        prompt += "\n\n"
    else:
        prompt += "No key events like passes or dribbles were recorded.\n\n"

    prompt += "--- ANALYSIS ---"
    return prompt

def generate_tactical_report(team_stats_df, player_stats_df, events_df):
    """
    Generates a tactical report using an LLM.

    Returns:
        str: The generated tactical report or an error message.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY environment variable not set. Cannot generate tactical report."

    openai.api_key = api_key

    formatted_prompt = format_data_for_llm(team_stats_df, player_stats_df, events_df)

    print("Sending data to LLM for tactical analysis...")

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a world-class football tactical analyst."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        report = response.choices[0].message.content
        return report
    except Exception as e:
        return f"ERROR: Failed to generate report from LLM. Details: {e}"