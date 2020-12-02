import os
import streamlit as st 
import pickle
# EDA Pkgs
import pandas as pd 
import matplotlib.pyplot as plt
# Viz Pkgs 
import seaborn as sns 

model= pickle.load(open('model_scores.p','rb'))
results = pd.DataFrame.from_dict(model, orient='index',
                                        columns=['RMSE', 'MAE','R2'])

results_lin = results[:1]

results_ran = results[1:2]

results_XG = results[2:3]

results_LS = results[3:4]
    
#restults_df = restults_df.sort_values(by='RMSE', ascending=False).reset_index()

def main():
	""" Covid ML Dataset Explorer """
	st.title("Datasets For ML Explorer with Streamlit")
	st.subheader("Covid dataset Analyser and Prediction")

	html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:30px;padding:10px">Streamlit is Awesome</p></div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	

	# Read Data
	df = pd.read_csv('covid_19_clean_complete.csv',parse_dates=False)
	df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)
	df_data = df.groupby(["Date", "Country", "Province/State"])[['Date', 'Province/State', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
	df1=df.groupby('Date').sum()
    
	max_case_count=pd.DataFrame(df1.groupby("Date")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index())
	
	# Show Dataset

	if st.checkbox("Show Dataset"):
		
		st.dataframe(df.head())

	# Show Columns
	if st.button("Column Names"):
		st.write(df.columns)

	# Show Shape
	if st.checkbox("Shape of Dataset"):
		data_dim = st.radio("Show Dimension By ",("Rows","Columns"))
		if data_dim == 'Rows':
			st.text("Number of Rows")
			st.write(df.shape[0])
		elif data_dim == 'Columns':
			st.text("Number of Columns")
			st.write(df.shape[1])
		else:
			st.write(df.shape)

	# Select Columns
	if st.checkbox("Select Columns To Show"):
		all_columns = df_data.columns.tolist()
		selected_columns = st.multiselect("Select",all_columns)
		new_df = df_data[selected_columns]
		st.dataframe(new_df)
	
	# Show Values
	if st.button("Value Counts"):
		st.text("Value Counts By Target/Class")
		st.write(df_data.iloc[:,-1].value_counts())


	# Show Datatypes
	if st.button("Data Types"):
		st.write(df_data.dtypes)



	# Show Summary
	if st.checkbox("Summary"):
		st.write(df_data.describe().T)
		
	# EDA
	
     
	if st.checkbox("Total cases"):
	    st.write(max_case_count)
	## Plot and Visualization

	st.subheader("Data Visualization")
	# Correlation
	# Seaborn Plot
	if st.checkbox("Correlation Plot[Seaborn]"):
		st.write(sns.heatmap(df_data.corr(),annot=True))
		st.pyplot()

	
	# Pie Chart
	if st.checkbox("Pie Plot"):
		all_columns_names = df_data.columns.tolist()
		if st.button("Generate Pie Plot"):
			st.success("Generating A Pie Plot")
			st.text("Total % of confrmed cases")
			st.write(df_data.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
			st.write(df_data.iloc[:,-2].value_counts().plot.pie(autopct="%1.1f%%"))
			st.write(df_data.iloc[:,-3].value_counts().plot.pie(autopct="%1.1f%%"))
			st.pyplot()

	
    
    # Count Plot
	if st.checkbox("Plot of Value Counts"):
		st.text("Value Counts By Target")
		all_columns_names = df_data.columns.tolist()
		primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
		selected_columns_names = st.multiselect("Select Columns",all_columns_names)
		if st.button("Plot"):
			st.text("Generate Plot")
			if selected_columns_names:
				vc_plot = df_data.groupby(primary_col)[selected_columns_names].count()
			else:
				vc_plot = df_data.iloc[:,-1].value_counts()
			st.write(vc_plot.plot(kind="bar"))
			st.pyplot()


	# Customizable Plot

	st.subheader("Customizable Plot")
	all_columns_names = df_data.columns.tolist()
	type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
	selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

	if st.button("Generate Plot"):
		st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

		# Plot By Streamlit
		if type_of_plot == 'area':
			cust_data = df_data[selected_columns_names]
			st.area_chart(cust_data)

		elif type_of_plot == 'bar':
			cust_data = df_data[selected_columns_names]
			st.bar_chart(cust_data)

		elif type_of_plot == 'line':
			cust_data = df_data[selected_columns_names]
			st.line_chart(cust_data)

		# Custom Plot 
		elif type_of_plot:
			cust_plot= df_data[selected_columns_names].plot(kind=type_of_plot)
			st.write(cust_plot)
			st.pyplot()

    # genrate model
    # Customizable Plot

	st.subheader("Customizable Plot")
	all_columns_names = df_data.columns.tolist()
	type_of_model = st.selectbox("Select Type of model",["LINEAR REGRASSION","RANDOM FOREST","XGBoost","LSTM"])
	#selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

	if st.button("Generate model"):
		#st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

		# Plot By Streamlit
		if type_of_model == 'LINEAR REGRASSION':
			#cust_data = df[selected_columns_names]
			st.write(results_lin)

		elif type_of_model == 'RANDOM FOREST':
			#cust_data = df[selected_columns_names]
			st.write(results_ran)

		elif type_of_model == 'XGBoost':
			#cust_data = df[selected_columns_names]
			st.write(results_XG)

		# Custom Plot 
		elif type_of_model:
			#cust_model= df_data[selected_columns_names].plot(kind=type_of_model)
			st.write(results_LS)
			#st.pyplot()

	if st.button("Models Results"):
		st.write(results)
	

	st.sidebar.header("About App")
	st.sidebar.info("A Simple EDA App for Exploring Common ML Dataset")

	st.sidebar.header("Get Datasets")
	st.sidebar.markdown("[Common ML Dataset Repo]("")")

	st.sidebar.header("About")
	st.sidebar.info("nitin faye@uttakarsh")
	st.sidebar.text("Built with Streamlit")
	st.sidebar.text("Maintained by Nitin Faye")


if __name__ == '__main__':
	main()

