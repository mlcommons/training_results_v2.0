// Copyright 2016 The Closure Rules Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

goog.module('goblin');

var dom = goog.require('goog.dom');


/**
 * Morning and evening
 * Maids heard the goblins cry:
 * “Come buy our orchard fruits,
 * Come buy, come buy:
 * Apples and quinces,
 * Lemons and oranges,
 * Plump unpeck’d cherries,
 * Melons and raspberries,
 * Bloom-down-cheek’d peaches,
 * Swart-headed mulberries,
 * Wild free-born cranberries,
 * Crab-apples, dewberries,
 * Pine-apples, blackberries,
 * Apricots, strawberries;—
 * All ripe together
 * In summer weather,—
 * Morns that pass by,
 * Fair eves that fly;
 * Come buy, come buy:
 * Our grapes fresh from the vine,
 * Pomegranates full and fine,
 * Dates and sharp bullaces,
 * Rare pears and greengages,
 * Damsons and bilberries,
 * Taste them and try:
 * Currants and gooseberries,
 * Bright-fire-like barberries,
 * Figs to fill your mouth,
 * Citrons from the South,
 * Sweet to tongue and sound to eye;
 * Come buy, come buy.”
 */
function goblin() {
  dom.getElement('goblin');
  console.log('Goblin Market by Christina Rossetti');
}


exports = goblin;
