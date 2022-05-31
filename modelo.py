import streamlit as st
import pandas as pd 
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title = 'Calcule o Salário dos Profissionais de Dados - FLAI', 
				   page_icon = 'iconeflai.png' ,
				   layout = 'centered', 
				   initial_sidebar_state = 'auto')

modelo = load_model('modelo-para-previsao-de-salario-2022')

#@st.cache
def ler_dados():
	dados = pd.read_csv('dataset-profissionais-dados-resumido.csv')
	dados = dados.dropna()
	return dados

dados = ler_dados()  

st.image('bannerflai.jpg', use_column_width = 'always')

st.write('''
# :sparkles: Modelo para Precificação de Salários para Profissionais de Dados
***Criado por [FLAI - Inteligência Artificial e Data Science](https://www.flai.com.br/)***. 

---

Nesse Web-App podemos utilizar um modelo de machine learning para estimar salários de profissionais da área de dados.

Entre com as características do profissional e da vaga, e verifique o valor estimado para o salário de mercado desse profissional. 

O modelo desse web-app foi desenvolvido utilizando o conjunto de 
dados que pode ser encontrado nesse [link do kaggle](https://www.kaggle.com/datahackers/pesquisa-data-hackers-2019).

''')

st.markdown('---') 
 
st.markdown('## Informações da vaga')
col1, col2 = st.columns(2)

x1 = col1.selectbox('Profissão', dados['Profissão'].unique().tolist())
x2 = col1.selectbox('Tamanho da Empresa', dados['TamanhoEmpresa'].unique().tolist()) 
x3 = col2.selectbox('Estado', dados['Estado'].unique().tolist()) 
x4 = col2.selectbox('Tipo de Trabalho', dados['TipoTrabalho'].unique().tolist() )
x5 = col2.selectbox('Setor de Mercado', dados['Setor'].unique().tolist())
x6 = col2.selectbox('Forma de Trabalho', dados['FormaTrabalho'].unique().tolist())
x7 = col2.selectbox('Nível', dados['Nivel'].unique().tolist()) 
x8 = col1.selectbox('Experiência em DS', dados['Experiência'].unique().tolist()) 
x9 = col1.selectbox('Área de Formação', dados['Formação'].unique().tolist())
x10 = col1.selectbox('Escolaridade', dados['Escolaridade'].unique().tolist()) 
x21 = col1.selectbox('Idade', dados['Idade'].unique().tolist()) 
x11 = col2.selectbox('Gestão', ['sim', 'não']) 


st.markdown('## Habilidades')
col1, col2 , col3 = st.columns(3)

x12 = col1.radio('Linguagem Python', dados['LinguagemPython'].unique().tolist()) 
x13 = col2.radio('Linguagem R', dados['LinguagemR'].unique().tolist()) 
x14 = col3.radio('Linguagem SQL', dados['LinguagemSQL'].unique().tolist()) 

x15 = col1.radio('Amazon Web Services', dados['NuvemAWS'].unique().tolist()) 
x16 = col2.radio('Google Cloud Plataform', dados['NuvemGCP'].unique().tolist()) 
x17 = col3.radio('Azure', dados['NuvemAzure'].unique().tolist()) 

x18 = col1.radio('Qlik View', dados['QlikView'].unique().tolist()) 
x19 = col2.radio('Power BI', dados['PowerBI'].unique().tolist()) 
x20 = col3.radio('Tableau', dados['Tableau'].unique().tolist()) 
 

dicionario  =  {'Profissão': [x1],
				'TamanhoEmpresa': [x2],
				'Estado': [x3],	
				'TipoTrabalho': [x4],
				'Setor': [x5],
				'FormaTrabalho': [x6],
				'Nivel': [x7],
				'Experiência': [x8],
				'Formação': [x9],
				'Escolaridade': [x10],
				'Gestão': [x11], 
				'Idade': [x21], 
				'LinguagemPython': [x12],
				'LinguagemR': [x12],
				'LinguagemSQL': [x14],
				'NuvemAWS': [x15],
				'NuvemGCP': [x16],
				'NuvemAzure': [x17],
				'QlikView': [x18],
				'PowerBI': [x19],
				'Tableau': [x20]}

dados = pd.DataFrame(dicionario)  

st.markdown('---') 

st.markdown('## Executar o Modelo de Precificação') 
 

if st.button('CALCULAR O SALÁRIO'):
	st.markdown('---') 
	saida = float(predict_model(modelo, dados)['Label']) 
	st.markdown('# Salário estimado de **R$ {:.2f}**'.format(saida))
	st.markdown('---') 

 
