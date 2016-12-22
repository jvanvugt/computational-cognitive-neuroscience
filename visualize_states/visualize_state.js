/**
 * Author: Joris van Vugt
 * 
 * Visualize the activation of each cell for each generated character
 * Idea stolen from Andrej Karpathy (http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
 * 
 * It expects a csv called cell_states.csv with all the activations (T, H) and a
 * file called generated.txt with all the generated text. These should be in the
 * same folder.
 * 
 */
let currentCell = 0;

function activationToColor(activation) {
    activation = Math.tanh(activation);
    if (activation <= 0) {
        let blue = Math.round(255 - Math.abs(activation) * 255);
        return `rgb(${blue}, ${blue}, 255)`;
    }
    let red = Math.round(255 - activation * 255);
    return `rgb(255, ${red}, ${red})`;
}

function parseCsv(csv) {
    let lines = csv.split('\n');
    let table = lines.map(line => line.split(',').map(parseFloat));
    return table;
}

fetch('cell_states.csv', {method: 'GET', mode: 'cors', cache: 'default'})
    .then(response => response.text())
    .then(csv => {
        let cellStates = parseCsv(csv);
        document.getElementById('total-cells').innerHTML = cellStates[0].length;
        fetch('generated.txt')
            .then(response => response.text())
            .then(text => {
                text = text.replace('\n', '\r\n');
                window.drawActivations = function() {
                    let htmlToWrite = '';
                    for (let i = 0; i < cellStates.length; i++) {
                        let letter = text[i].replace('\r', ' ').replace('\n', ' \n');
                        let color = activationToColor(cellStates[i][currentCell]);
                        htmlToWrite += `<span style="background-color:${color}">${letter}</span>`;
                    }
                    let text_div = document.getElementById('text');
                    text_div.innerHTML = htmlToWrite;
                }
                window.drawActivations();
            });
    });

function nextCell() {
    currentCell++;
    document.getElementById('current-cell').innerHTML = currentCell;
    window.drawActivations();
}

function previousCell() {
    currentCell--;
    document.getElementById('current-cell').innerHTML = currentCell;
    window.drawActivations();
}

function gotoCell(e) {
    currentCell = e.value;
    document.getElementById('current-cell').innerHTML = currentCell;
    window.drawActivations();
}

window.addEventListener('keydown', (e) => {
    if (e.keyCode == 39) {
        nextCell();
    }
    else if (e.keyCode == 37) {
        previousCell();
    }
});
