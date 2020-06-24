package main
import (
	"fmt"
	"log"
	"net/http"
	"github.com/gorilla/mux"
	"io/ioutil"
	"encoding/json"
	"math/rand"
	"math"
	"gonum.org/v1/gonum/floats"
)

//Matrix matrix-type
type Matrix [][]float64

type schedule struct {
	Path    []int `json:"path"` 
	ServiceTime    []float64 `json:"service_time"`
	TwBegin    []float64 `json:"tw_begin"`
	TwEnd    []float64 `json:"tw_end"`
	CostMatrix Matrix  `json:"cost_matrix"`
	StdMatrix Matrix  `json:"std_matrix"`
	Iterations int  `json:"iterations"`
}
type assignment []schedule

type solvedAssignment struct {
	P	[]int     
	S	float64  
	IDX int
	SSamples [][]float64 
	PSamples [][][]int
}

type fitIn struct {
	P	[][][]int     
	S	[][]float64  
	IDX int
}


type payload struct {
	P	[][]int    `json:"path"` 
	S	[]float64  `json:"score"` 
	IDX []int	   `json:"idx"` 
	Samples [][][][]int `json:"samples"` 
	SampleScores [][][]float64 `json:"samples_scores"` 
}

//ClampToZero clamps negative values to 0
func ClampToZero(x int) int {
	if x <= 0 {
		return 0
	}else{
		return x
	}
}

//Rows get rows of matrix type
func Rows(M Matrix) int{return len(M)}

//Cols get cols of matrix type
func Cols(M Matrix) int{return len(M[0])}

//InitMatrix create matrix with zeros
func InitMatrix(rows, cols int) Matrix {
	M := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		M[i] = make([]float64, cols)
	}
	return M
}

//DeepCopyMatrix Create a deep copy of any matrix where src and dst have same dims
func DeepCopyMatrices(dstTrans Matrix, dstStoc Matrix, srcTrans Matrix, srcCost Matrix, srcStd Matrix) (Matrix, Matrix) {
	rows := Rows(srcTrans)
	for i := 0; i < rows; i++ { 
		stocEntry := make([]float64, rows)
		for j := 0; j < rows; j++ {
			a :=  srcCost[i][j] + rand.NormFloat64()*srcStd[i][j]
			if a < 0 {a = 0}
			stocEntry[j] = a
		}

		copy(dstTrans[i], srcTrans[i])
		copy(dstStoc[i], stocEntry) 
	}
	return dstTrans, dstStoc
}

//InitProbMatrix Load matrix with symmetric unnormalized probs and zero diagonal 
func InitProbMatrix(M Matrix) Matrix{
	rows := Rows(M)
	cols := Rows(M)
	for j := 0; j < cols; j++ {
		for i := (j+1); i < rows; i++ {
			M[j][i] = 1
			M[i][j] = 1
		}
	}
	return M
}

//WeightedPick Pick index based on unnormalized weights
func WeightedPick(probs []float64) int {
	partition := make([]float64, len(probs))
	floats.CumSum(partition, probs)
	norm := partition[len(partition)-1]
	normPartition := make([]float64, len(probs))

	for i, val := range partition {normPartition[i] = val/norm}
	rUniform := rand.Float64()

	for q:=0; q < len(normPartition); q++ {
		if rUniform <= normPartition[q]{
			return q
		}
	}
	return -1
}

//CloseTransition Zero rows and columns in transition based on visited node index
func CloseTransition(idx int, M Matrix) Matrix {
	cols := Rows(M)
	for j := 0; j < cols; j++ {
		M[j][idx] = 0.0
	}
	return M
} 

//McSampler generate random routes for monte carlo sampling
func McSampler(InitTransitionMatrix Matrix, costMatrix Matrix, stdMatrix Matrix, serviceTimes []float64, twBegin []float64, twEnd []float64, samples int, mode int) ([][]int, []float64) {	
	result := [][]int{}
	scores := []float64{}

	rows := Rows(costMatrix)
	cols := Cols(costMatrix)

	for s:=0; s < samples; s++ {
		idx := 0
		iterations := 0
		pathScore := 0.0

		transitionMatrix := InitMatrix(rows, cols)
		stochasticCostMatrix := InitMatrix(rows, cols) 

		transitionMatrix, stochasticCostMatrix  = DeepCopyMatrices(transitionMatrix, stochasticCostMatrix, InitTransitionMatrix, costMatrix, stdMatrix)
		
		samplePath := []int{}
		samplePath = append(samplePath, 0)

		if mode == 0 {transitionMatrix = CloseTransition(0, transitionMatrix)}

		for idx != -1{
			twTransitionProbs := CloseInvalidTransitions(samplePath, stochasticCostMatrix , transitionMatrix, serviceTimes, twBegin, twEnd)
			lastIdx := idx
			idx = WeightedPick(twTransitionProbs)
			samplePath = append(samplePath, idx)
			pathScore = pathScore + stochasticCostMatrix[ClampToZero(idx)][lastIdx]
			
			if idx == -1 || idx == 0 {
				break
			}
			transitionMatrix = CloseTransition(idx, transitionMatrix)
			iterations++
			if iterations > Rows(transitionMatrix){
				break
			}
		}
		result = append(result, samplePath)
		scores = append(scores, pathScore)
	}
	return result, scores
}

//Quantile Sort the sample paths and get the nth quantile of highest scoring samples
func Quantile(quantile float64, values []float64, samplePaths [][]int) ([]float64, [][]int, []int) {
	sortedPaths := [][]int{}
	sortedValues := []float64{}
	sortedIndices := []int{}
	indices := []int{}

	for i := 0; i < len(values); i++ { 
		indices = append(indices, i)
	 }

	nQuantileIndices := int(math.Ceil(quantile*float64(len(values))))
	floats.Argsort(values, indices)

	for i := 0; i < nQuantileIndices; i++{
		sortedPaths = append(sortedPaths, samplePaths[indices[i]])
		sortedValues = append(sortedValues, values[i])
		sortedIndices = append(sortedIndices, indices[i])
	}
	return sortedValues, sortedPaths, sortedIndices
}

//AccumulateRoute get travel time of partial or completed routes
func GetAccumulatedRoute(partialRoute []int, costMatrix Matrix, serviceTimes []float64) float64 {
	if len(partialRoute) < 2 {return 0.0}
	if Rows(costMatrix) != len(serviceTimes){panic("Service time vector should have the same dimentions as the cost matrix")}
	time := 0.0
	for i, startIdx := range partialRoute[:len(partialRoute)-1] {
		endIdx := partialRoute[i+1]
		if endIdx != -1{
			time += costMatrix[startIdx][endIdx]
		}else{
			time += costMatrix[startIdx][0] 
		}
		time += serviceTimes[startIdx]
	}
	return time
}

//CloseInvalidTransitions Close transitions that are impossible either as a consequence of time window constraints or locked assignments
func CloseInvalidTransitions(partialRoute []int, costMatrix Matrix, transitionMatrix Matrix, serviceTimes []float64, twBegin []float64, twEnd []float64) []float64  {
	startIdx := partialRoute[len(partialRoute)-1]
	updatedTransitions := make([]float64, Rows(costMatrix)) 
	
	for endIdx := range updatedTransitions{
		partialTime := GetAccumulatedRoute(partialRoute, costMatrix, serviceTimes)
		transitionTime := transitionMatrix[startIdx][endIdx] + serviceTimes[endIdx]
		jobEndsAt := partialTime + transitionTime

		if twEnd[endIdx] < jobEndsAt || twEnd[0] < jobEndsAt {
			updatedTransitions[endIdx] = 0
		}else{
			updatedTransitions[endIdx] = transitionMatrix[startIdx][endIdx]
		}

		//if partialTime < twBegin[endIdx]{
		//	updatedCost[endIdx] += math.Abs(twBegin[endIdx] - partialTime)
		//}
	}
	return updatedTransitions
}

//UpdateTransitionMatrix increase probability of neat transitions
func UpdateTransitionMatrix(transitionMatrix Matrix, quantileValues []float64, quantilePaths [][]int, smoother float64) Matrix{
	for _, path:= range quantilePaths{
		for j, startIdx:= range path {
			endIdx := path[j+1]
			if endIdx != -1{
				transitionMatrix[startIdx][endIdx] += smoother
			}else{
				transitionMatrix[startIdx][0] += smoother
			}
			if endIdx == -1 {break}
		} 
	}
	return transitionMatrix
}

//ProcessRequest get request and solve assignment
func ProcessRequest(w http.ResponseWriter, r *http.Request) {
	newAssignment := assignment{}
	reqBody, err := ioutil.ReadAll(r.Body)
	mode := mux.Vars(r)["mode"]

	if err != nil {
		fmt.Fprintf(w, "Kindly enter data with the event title and description only in order to update")
	}
	json.Unmarshal(reqBody, &newAssignment)
	DRIVERS := len(newAssignment)
	SCORES := []float64{}
	ROUTES := [][]int{}
	IDX := []int{}
	PSamples := [][][][]int{}
	SSamples := [][][]float64{}

	C0 := make(chan solvedAssignment)
	C1 := make(chan fitIn)
	for d := 0; d < DRIVERS; d++ {
		costMatrix := newAssignment[d].CostMatrix
		stdMatrix := newAssignment[d].StdMatrix

		serviceTime := newAssignment[d].ServiceTime
		twBegin := newAssignment[d].TwBegin
		twEnd := newAssignment[d].TwEnd
		iterations := newAssignment[d].Iterations
		path := newAssignment[d].Path

		rows := Rows(costMatrix)
		cols := Cols(costMatrix)

		transitionMatrix := InitMatrix(rows, cols)
		transitionMatrix = InitProbMatrix(transitionMatrix)
	
		if mode == "0" {
			go SolverMode0(C0, iterations, transitionMatrix, costMatrix, stdMatrix, serviceTime, twBegin, twEnd, d)
			result := <- C0
			SCORES = append(SCORES, result.S)
			ROUTES = append(ROUTES, result.P)
			SSamples = append(SSamples, result.SSamples)
			PSamples = append(PSamples, result.PSamples)

			IDX = append(IDX, result.IDX)
		}

		if mode == "1"{
			go SolverMode1(iterations, C1, path, costMatrix, stdMatrix, serviceTime, twBegin, twEnd, d)
			result := <-C1
			returnPath := EvaluateScore(result, 0.9)
			ROUTES = append(ROUTES, returnPath)
			PSamples = append(PSamples, result.P)
			SSamples = append(SSamples, result.S)
			IDX = append(IDX, result.IDX)
		}
	}

	payload := payload{
		S: SCORES, 
		P: ROUTES,
		Samples: PSamples,
		SampleScores: SSamples,
		IDX: IDX,
	}
	
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(payload)	
}

//SolverMode0 Keep all jobs fixed, and fit in any additional jobs
func SolverMode0(C chan solvedAssignment, iterations int, transitionMatrix Matrix, costMatrix Matrix, stdMatrix Matrix, serviceTime []float64, twBegin []float64, twEnd []float64, d int) {
	scoreList := make([]float64, 2)
	finalRoute := []int{}
	pathSamples := [][][]int{}
	scoreSamples := [][]float64{}

	for i := 0; i < iterations; i++ {
		samples, scores := McSampler(transitionMatrix, costMatrix, stdMatrix, serviceTime, twBegin, twEnd, 150, 0)
		quantileValues, quantilePaths, _ := Quantile(0.025, scores, samples)
		
		scoreSamples = append(scoreSamples, quantileValues)
		pathSamples = append(pathSamples, quantilePaths)

		transitionMatrix = UpdateTransitionMatrix(transitionMatrix, quantileValues, quantilePaths, 1.0)
		finalRoute = quantilePaths[len(quantilePaths)-1]
		score := floats.Sum(quantileValues)
		scoreList = append(scoreList, score)
		if scoreList[len(scoreList)-1] == scoreList[len(scoreList)-3]{break}
	}
	score := scoreList[len(scoreList)-1]
	solvedAssignment := solvedAssignment{
		S: score,
		P: finalRoute,
		IDX: d,
		SSamples: scoreSamples,
		PSamples: pathSamples,
	}
	C <- solvedAssignment
}

//SolverModel1 Solve in mode 1
func SolverMode1(samples int, C chan fitIn, path []int, costMatrix Matrix, stdMatrix Matrix, serviceTime []float64, twBegin []float64, twEnd []float64, d int) {
	score := [][]float64{}
	insertPath := [][][]int{}
	rows := Rows(costMatrix)
	cols := Cols(costMatrix)
	
	for s:=0; s < samples; s++ {
		score = append(score, []float64{})
		insertPath = append(insertPath, [][]int{})
		transitionMatrix := InitMatrix(rows, cols)
		stochasticCostMatrix := InitMatrix(rows, cols) 
		_, stochasticCostMatrix  = DeepCopyMatrices(transitionMatrix, stochasticCostMatrix, costMatrix, costMatrix, stdMatrix)

		for i := 0; i < (len(path)-1); i++ {
			startSlot := GetAccumulatedRoute(path[:i], stochasticCostMatrix, serviceTime)
			endSlot := GetAccumulatedRoute(path[:(i+1)], stochasticCostMatrix, serviceTime)
	
			modPath := []int{}
			cpyBegin := make([]int, len(path[:(i+1)]))
			copy(cpyBegin, path[:(i+1)])
			cpyEnd := make([]int, len(path[(i+1):len(path)-1]))
			copy(cpyEnd, path[(i+1):])

			modPath = cpyBegin
			modPath = append(modPath, path[len(path)-1])
			modPath = append(modPath, cpyEnd...)
			insertPath[s] = append(insertPath[s] , modPath)
	
			dt := endSlot - startSlot
			travelToFromNew := 0.0

			if (i+1) > (len(path) - 1){
				travelToFromNew = serviceTime[len(serviceTime)-1] + stochasticCostMatrix[path[i]][len(path)-1]  + stochasticCostMatrix[path[0]][len(path)-1] 
			}else{
				travelToFromNew = serviceTime[len(serviceTime)-1] + stochasticCostMatrix[path[i]][len(path)-1]  + stochasticCostMatrix[path[i+1]][len(path)-1]
			}

			if dt > travelToFromNew && startSlot >= twBegin[len(twBegin)-1] && endSlot <= twEnd[len(twEnd)-1]{
				score[s] = append(score[s], dt - travelToFromNew)
			}else{
				score[s] = append(score[s], -1)
			}
		}
	}
	solvedAssignment := fitIn{
		S: score,
		P: insertPath,
		IDX: d,
	}
	C <- solvedAssignment		
}

//EvaluateScore - For mode 1. Pick the most ideal path
func EvaluateScore(solvedAssignment fitIn, tol float64) []int {
	rejectedPaths := make([]float64, len(solvedAssignment.S[0]))
	totalScore := make([]float64, len(solvedAssignment.S[0]))
	indices := make([]int, len(solvedAssignment.S[0]))
	for s := 0; s < len(solvedAssignment.S[0]); s++{
		indices[s] = s
		for i := 0; i < len(solvedAssignment.S[s]); i++{
			if solvedAssignment.S[s][i] == -1{
				rejectedPaths[i] += 1.0/float64(len(solvedAssignment.S))
			}
			if rejectedPaths[i] < tol && solvedAssignment.S[s][i] > -1 {
				totalScore[i] += solvedAssignment.S[s][i]
			}else{
				totalScore[i] = 99999999
			}	
		}
	}
	floats.Argsort(totalScore, indices)
	if totalScore[0] != 99999999 {
		return solvedAssignment.P[0][indices[0]]
	}else{
		fail := make([]int, 1)
		fail[0] = -1
		return fail
	}
}

func main() {
	router := mux.NewRouter().StrictSlash(true)
	router.HandleFunc("/solve/{mode}", ProcessRequest)
	log.Fatal(http.ListenAndServe(":8080", router))
}
