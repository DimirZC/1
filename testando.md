importar  numpy  como  np
importar  matplotlib . pyplot  como  plt
importar  pandas  como  pd
importar  pandas_datareader  como  web
importar  data e hora  como  dt

de  sklearn . importação de pré -processamento  MinMaxScaler 
do  tensorflow . . _ importação de modelos  sequencial 
do  tensorflow . . _ Importação de camadas  Denso , Dropout , LSTM 

#%%
#Importando Dados

empresa  =  'FB'

início  =  dt . data e hora ( 2012 , 1 , 1 )
final  =  dt . data e hora ( 2020 , 1 , 1 )

dados  =  web . DataReader ( empresa , 'yahoo' , inicio , final )

#Preparar Dados
normalizando  =  MinMaxScaler ( feature_range = ( 0 , 1 ))
dados_normalizados  =  normalizando . fit_transform ( dados [ 'Fechar' ]. valores . reshape ( -1 , 1 ) )

previsão_dias  =  60

x_treinar , y_treinar  = [], []

for  x  in  range ( previsao_dias , len ( dados_normalizados )):
    x_treinar . append ( dados_normalizados [ x - previsão_dias : x , 0 ])
    y_treinar . append ( dados_normalizados [ x , 0 ])
    
x_treinar , y_treinar  =  np . matriz ( x_treinar ), np . matriz ( y_treinar )
x_treinar  =  np . remodelar ( x_treinar , ( x_treinar . forma [ 0 ], x_treinar . forma [ 1 ], 1 ))


#Construindo nosso modelo de rede neural
modelo  =  Sequencial ()

. _ add ( LSTM ( unidades = 50 , return_sequences = True , input_shape = ( x_treinar . shape [ 1 ], 1 )))
. _ adicionar ( Desistência ( 0,2 ))
. _ add ( LSTM ( unidades = 50 , return_sequences = True ))
. _ adicionar ( Desistência ( 0,2 ))
. _ adicionar ( LSTM ( unidades = 50 ))
. _ adicionar ( Desistência ( 0,2 ))
. _ add ( Dense ( units  =  1 )) #Prevendo o proximo valor da acao

. _ compile ( otimizador  =  'adam' , perda  =  'mean_squared_error' )
. _ fit ( x_treinar , y_treinar , épocas  =  25 , batch_size  =  32 )

###Testando a precisão do nosso modelo em dados existentes

#preparando alguns dados para testar
teste_inicio  =  dt . data e hora ( 2020 , 1 , 1 )
teste_final  =  dt . datahora . agora ()

dados_teste  =  web . DataReader ( empresa , 'yahoo' , teste_inicio , teste_final )
precos_reais  =  dados_teste [ 'Fechar' ]. valores

total_dados  =  pd . concat (( dados [ 'Fechar' ], dados_teste [ 'Fechar' ]), axis  =  0 )

modelo_entrada  =  total_dados [ len ( total_dados ) -  len ( dados_teste ) -  previsao_dias :]. valores
modelo_entrada  =  modelo_entrada . remodelar ( - 1 , 1 )
modelo_entrada  =  normalizando . transform ( modelo_entrada )


#Fazer previsoes nos valores de teste

x_teste  = []

for  x  in  range ( previsao_dias , len ( modelo_entrada )):
    x_teste . append ( modelo_entrada [ x - previsão_dias : x , 0 ])
    
x_teste  =  np . matriz ( x_teste )
x_teste  =  np . remodelar ( x_teste , ( x_teste . forma [ 0 ], x_teste . forma [ 1 ], 1 ))

previsão_precos  =  modelo . prever ( x_teste )
previao_precos  =  normalizando . inverse_transform ( previsão_precos )

#Representando Graficamente as Previsões
plt . plot ( precos_reais , color  = 'red' , label  =  f"Valor Real das ações de { empresa } " )
plt . plot ( previsão_precos , color = "green" , label  =  f"Previsão das ações de { empresa } " )
plt . title ( f" { empresa } Preço Ação" )
plt . xlabel ( 'Tempo' )
plt . ylabel ( f" { empresa } Preço Ação" )
plt . legenda ()
plt . mostrar ()


#%%
#Prevendo os próximos dias

dados_reais  =   [ modelo_entrada [ len ( modelo_entrada ) +  1  -  previsao_dias : len ( modelo_entrada  +  1 ), 0 ]]
dados_reais  =   np . matriz ( dados_reais )
dados_reais  =  np . reshape ( dados_reais , ( dados_reais . forma [ 0 ], dados_reais . forma [ 1 ], 1 ))

previsão  =  modelo . prever ( dados_reais )
previao  =  normalizando . inverse_transform ( previsão )

print ( f"Previsão para amanha: { previsão } " )
