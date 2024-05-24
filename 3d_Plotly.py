graficacion 3d interactiva con plotly
recomiendo mas usar plotly para graficar y asi obtener una grafica deinamica como la siguiente:

el codigo es el siguiente:
Nota: no realice el paso de escalamiento(para mas facilidad)

#LIBs
import plotly.express as px
import plotly.graph_objects as go

# TRAINING MODEL
X = df[['RM', 'INDUS']].values
y = df['MEDV'].values.reshape(-1, 1)


slr = LinearRegression()
slr.fit(X, y)

# PLOTTING
mesh_size = .02
margin = 0

# Create a mesh grid on which we will run our model
x_min, x_max = X[:,0].min() - margin, X[:,0].max() + margin
y_min, y_max = X[:,1].min() - margin, X[:,1].max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Run model
pred = slr.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)

# Generate the plot
fig = px.scatter_3d(df, x='RM', y='INDUS', z='MEDV')
fig.update_traces(marker=dict(size=5))
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
fig.show()
