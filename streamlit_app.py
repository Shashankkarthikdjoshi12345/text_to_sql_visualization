import streamlit as st
import subprocess
import os
import time
from langchain_core.messages import HumanMessage
from main import app  # Import the LangGraph workflow from main.py
from PIL import Image
import textwrap

st.title("SQL Query & Visualization App")

# User input
user_query = st.text_area(
    "Enter your query:", 
    "List all customers and their total order amount, exclude those who haven't placed any orders. Plot as Pie Chart"
)

if st.button("Run Query"):
    if user_query.strip():
        # Prepare the query for LangGraph workflow
        query = {"messages": [HumanMessage(content=user_query)]}

        response = app.invoke(query)

        final_output = response["messages"][-1].content
        # st.subheader("Query Result:")
        # st.write(final_output)
        st.subheader("Query Result:")
        pretty_output = textwrap.fill(final_output, width=80)  # Adjust width for better readability
        st.text_area("", pretty_output, height=200)



        # Run the generated `plot.py` script
        plot_path = "generated_plot.png"

        if os.path.exists("plot.py"):
            try:
                # Execute the plot script
                subprocess.run(["python", "plot.py"], check=True)
                time.sleep(2)  # Allow time for the plot to be generated

                # Check if the plot image was created
                if os.path.exists(plot_path):
                    st.subheader("Generated Visualization:")
                    st.image(plot_path, use_container_width=True)
                else:
                    st.error("Plot generation failed. No output image found.")
                os.remove("/Users/shjoshi/Desktop/AGENTIC_AI_POC_TEAMS/plot.py")
            except Exception as e:
                st.error(f"Error running plot script: {e}")
        else:
            # st.error("plot.py not found! Ensure the script is generated before execution.")
            pass

    else:
        st.warning("Please enter a valid query.")
